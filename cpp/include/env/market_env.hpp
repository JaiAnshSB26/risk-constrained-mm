// ============================================================================
// risk-constrained-mm :: cpp/include/env/market_env.hpp
// ============================================================================
// MarketEnvironment — the C++ core of the Gymnasium RL environment.
//
// Architecture:
//   ┌──────────────────────────────────────────────────────┐
//   │                  MarketEnvironment                    │
//   │                                                      │
//   │  ┌──────────┐   ┌────────────┐   ┌──────────────┐   │
//   │  │OrderBook │   │ Hawkes     │   │ obs_buffer_  │   │
//   │  │(matching)│   │ Simulator  │   │ (flat f64)   │   │
//   │  └──────────┘   └────────────┘   └──────────────┘   │
//   │                                                      │
//   │  Agent state: inventory_, pnl_, agent_bid/ask_ids_   │
//   └──────────────────────────────────────────────────────┘
//
// The observation buffer is pre-allocated ONCE and reused every step() —
// giving pybind11 a zero-copy view into the flat C array via array_t.
//
// Observation layout (4*N + 2 doubles):
//   [bid_price_0, bid_vol_0, ..., bid_price_{N-1}, bid_vol_{N-1},
//    ask_price_0, ask_vol_0, ..., ask_price_{N-1}, ask_vol_{N-1},
//    normalised_inventory, normalised_pnl]
//
// Prices are normalised: (price - mid) / tick_size.
// Volumes are normalised: total_qty / max_volume_norm.
// ============================================================================
#pragma once

#include "lob/order_book.hpp"
#include "lob/types.hpp"
#include "sim/hawkes_params.hpp"
#include "sim/hawkes_simulator.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace rcmm {

// ── Configuration ───────────────────────────────────────────────────────────

struct EnvConfig {
    // Book configuration.
    Price       tick_size    = 1;
    Price       base_price   = 49900;
    std::size_t num_levels   = 256;     // price-level array size

    // Observation.
    std::size_t obs_depth    = 5;       // top N levels per side in obs

    // Simulation.
    std::size_t ticks_per_step = 10;    // market ticks replayed per step()
    std::size_t max_steps      = 10000; // episode length
    std::size_t warmup_ticks   = 200;   // initial ticks to seed the book

    // Normalisation.
    double max_inventory  = 100.0;  // for obs normalisation
    double max_pnl        = 1e6;    // for obs normalisation
    double max_volume     = 100.0;  // volume normalisation denominator

    // Hawkes simulator.
    HawkesParams hawkes_params = NORMAL_REGIME;
    MarkConfig   mark_config   = {};
    std::uint64_t seed         = 42;
};

// ── Step result ─────────────────────────────────────────────────────────────

struct StepResult {
    const double* obs_data = nullptr;   // pointer into obs_buffer_
    std::size_t   obs_size = 0;
    double        reward   = 0.0;
    bool          done     = false;
    double        inventory = 0.0;
    double        pnl       = 0.0;
    std::size_t   step_num  = 0;
    std::size_t   fills     = 0;
};

// ── MarketEnvironment ───────────────────────────────────────────────────────

/// PoolCap for the embedded OrderBook — 64k slots is sufficient for sim.
inline constexpr std::size_t ENV_POOL_CAP = 1u << 16;

class MarketEnvironment {
public:
    explicit MarketEnvironment(EnvConfig cfg = {})
        : cfg_(cfg),
          book_(make_book_config(cfg)),
          sim_(cfg.hawkes_params, cfg.mark_config, cfg.seed),
          obs_buffer_(obs_size(), 0.0) {
        // Pre-generate a full episode's worth of ticks.
        regenerate_ticks();
    }

    // ── Observation size ────────────────────────────────────────────────────

    [[nodiscard]] std::size_t obs_size() const noexcept {
        return cfg_.obs_depth * 4 + 2;   // 2 sides × N × {price,vol} + inv + pnl
    }

    // ── Reset ───────────────────────────────────────────────────────────────

    /// Reset the environment to a fresh episode.
    /// Returns a pointer to the observation buffer (obs_size() doubles).
    const double* reset() {
        // Regenerate ticks for a new episode.
        regenerate_ticks();

        // Reset order book — we need a fresh one.
        book_.~OrderBook();
        new (&book_) OrderBook<ENV_POOL_CAP>(make_book_config(cfg_));

        // Reset agent state.
        inventory_    = 0.0;
        pnl_          = 0.0;
        step_count_   = 0;
        tick_cursor_  = 0;
        next_agent_id_ = AGENT_ID_BASE;
        agent_bid_id_ = 0;
        agent_ask_id_ = 0;

        // Warm up the book with initial ticks.
        warmup();

        // Fill observation buffer.
        fill_obs();

        return obs_buffer_.data();
    }

    // ── Step ────────────────────────────────────────────────────────────────

    /// Execute one step:
    ///   1. Cancel agent's previous quotes.
    ///   2. Place new quotes at (mid ± spread) with given sizes.
    ///   3. Advance simulation by ticks_per_step market ticks.
    ///   4. Detect fills on agent orders, update inventory & PnL.
    ///   5. Fill observation buffer, compute reward.
    StepResult step(double bid_spread, double ask_spread,
                    double bid_size, double ask_size) {
        ++step_count_;

        // 1. Cancel previous agent quotes.
        cancel_agent_quotes();

        // 2. Compute mid price and place new quotes.
        Price mid = compute_mid();
        auto bid_ticks = static_cast<Price>(
            std::max(1.0, std::round(bid_spread)));
        auto ask_ticks = static_cast<Price>(
            std::max(1.0, std::round(ask_spread)));
        Price bid_price = mid - bid_ticks * cfg_.tick_size;
        Price ask_price = mid + ask_ticks * cfg_.tick_size;

        auto bid_qty = static_cast<Qty>(std::max(1.0, std::round(bid_size)));
        auto ask_qty = static_cast<Qty>(std::max(1.0, std::round(ask_size)));

        // Clamp prices to book range.
        bid_price = std::clamp(bid_price, cfg_.base_price,
                               cfg_.base_price +
                               static_cast<Price>(cfg_.num_levels - 1) *
                               cfg_.tick_size);
        ask_price = std::clamp(ask_price, cfg_.base_price,
                               cfg_.base_price +
                               static_cast<Price>(cfg_.num_levels - 1) *
                               cfg_.tick_size);

        // Place agent limit orders.
        OrderId bid_id = next_agent_id_++;
        OrderId ask_id = next_agent_id_++;

        auto bid_trades = book_.place_order(
            bid_id, bid_price, bid_qty, Side::Bid, OrderType::Limit);
        auto ask_trades = book_.place_order(
            ask_id, ask_price, ask_qty, Side::Ask, OrderType::Limit);

        agent_bid_id_ = bid_id;
        agent_ask_id_ = ask_id;

        // Count immediate fills from placement.
        std::size_t fill_count = 0;
        process_agent_trades(bid_trades, Side::Bid, fill_count);
        process_agent_trades(ask_trades, Side::Ask, fill_count);

        // 3. Advance simulation: replay ticks_per_step market events.
        for (std::size_t i = 0; i < cfg_.ticks_per_step; ++i) {
            if (tick_cursor_ >= ticks_.size()) break;

            const auto& tick = ticks_[tick_cursor_++];
            auto trades = replay_tick(tick);

            // 4. Check if any trade involved our agent orders.
            for (const auto& t : trades) {
                if (t.maker_id == agent_bid_id_) {
                    // Agent's bid was lifted — we bought.
                    inventory_ += static_cast<double>(t.qty);
                    pnl_ -= static_cast<double>(t.price * t.qty);
                    ++fill_count;
                } else if (t.maker_id == agent_ask_id_) {
                    // Agent's ask was hit — we sold.
                    inventory_ -= static_cast<double>(t.qty);
                    pnl_ += static_cast<double>(t.price * t.qty);
                    ++fill_count;
                }
            }
        }

        // 5. Fill observation and compute reward.
        fill_obs();

        bool done = (step_count_ >= cfg_.max_steps) ||
                    (tick_cursor_ >= ticks_.size());

        // Reward: PnL change + inventory penalty.
        double mid_d = static_cast<double>(compute_mid());
        double mark_to_market = pnl_ + inventory_ * mid_d;
        double inv_penalty = -0.01 * inventory_ * inventory_;
        double reward = mark_to_market + inv_penalty - prev_mtm_;
        prev_mtm_ = mark_to_market + inv_penalty;

        StepResult result;
        result.obs_data  = obs_buffer_.data();
        result.obs_size  = obs_buffer_.size();
        result.reward    = reward;
        result.done      = done;
        result.inventory = inventory_;
        result.pnl       = pnl_;
        result.step_num  = step_count_;
        result.fills     = fill_count;

        return result;
    }

    // ── Accessors ───────────────────────────────────────────────────────────

    [[nodiscard]] const EnvConfig& config() const noexcept { return cfg_; }
    [[nodiscard]] double inventory() const noexcept { return inventory_; }
    [[nodiscard]] double pnl() const noexcept { return pnl_; }
    [[nodiscard]] std::size_t step_count() const noexcept { return step_count_; }
    [[nodiscard]] Price best_bid() const noexcept { return book_.best_bid(); }
    [[nodiscard]] Price best_ask() const noexcept { return book_.best_ask(); }
    [[nodiscard]] const double* obs_data() const noexcept { return obs_buffer_.data(); }

private:
    // Agent order IDs start well above any simulator-generated IDs.
    static constexpr OrderId AGENT_ID_BASE = 10'000'000;

    // ── Helpers ─────────────────────────────────────────────────────────────

    static BookConfig make_book_config(const EnvConfig& cfg) noexcept {
        return BookConfig{
            .tick_size  = cfg.tick_size,
            .base_price = cfg.base_price,
            .num_levels = cfg.num_levels,
            .pool_cap   = ENV_POOL_CAP,
        };
    }

    void regenerate_ticks() {
        std::size_t total = cfg_.warmup_ticks +
                            cfg_.max_steps * cfg_.ticks_per_step + 1000;
        // Re-seed for a new episode with varied randomness.
        sim_.seed(cfg_.seed + static_cast<std::uint64_t>(episode_count_++));
        ticks_ = sim_.simulate(total, 0);
        tick_cursor_ = 0;
    }

    void warmup() {
        for (std::size_t i = 0; i < cfg_.warmup_ticks && tick_cursor_ < ticks_.size(); ++i) {
            (void)replay_tick(ticks_[tick_cursor_++]);
        }
        prev_mtm_ = 0.0;
    }

    void cancel_agent_quotes() noexcept {
        if (agent_bid_id_ != 0) {
            (void)book_.cancel_order(agent_bid_id_);
            agent_bid_id_ = 0;
        }
        if (agent_ask_id_ != 0) {
            (void)book_.cancel_order(agent_ask_id_);
            agent_ask_id_ = 0;
        }
    }

    [[nodiscard]] Price compute_mid() const noexcept {
        Price bb = book_.best_bid();
        Price ba = book_.best_ask();
        // If either side is missing, fall back to the mark config mid.
        if (bb == std::numeric_limits<Price>::min() ||
            ba == std::numeric_limits<Price>::max()) {
            return cfg_.mark_config.mid_price;
        }
        return (bb + ba) / 2;
    }

    /// Replay a single tick on the book.  Returns any trades generated.
    [[nodiscard]] std::vector<Trade> replay_tick(const Tick& tick) {
        std::vector<Trade> trades;

        switch (tick.action) {
        case TickAction::Add:
            // Only add if price is within book range.
            if (tick.price >= cfg_.base_price &&
                price_in_range(tick.price)) {
                (void)book_.add_order(
                    tick.order_id, tick.side, tick.price, tick.qty,
                    tick.timestamp);
            }
            break;

        case TickAction::Cancel:
            (void)book_.cancel_order(tick.order_id);
            break;

        case TickAction::Modify:
            (void)book_.cancel_order(tick.order_id);
            if (tick.price >= cfg_.base_price &&
                price_in_range(tick.price)) {
                (void)book_.add_order(
                    tick.order_id, tick.side, tick.price, tick.qty,
                    tick.timestamp);
            }
            break;

        case TickAction::Trade:
            if (tick.price >= cfg_.base_price &&
                price_in_range(tick.price)) {
                trades = book_.place_order(
                    tick.order_id, tick.price, tick.qty, tick.side,
                    OrderType::Market, tick.timestamp);
            }
            break;
        }

        return trades;
    }

    [[nodiscard]] bool price_in_range(Price p) const noexcept {
        auto idx = static_cast<std::size_t>(
            (p - cfg_.base_price) / cfg_.tick_size);
        return idx < cfg_.num_levels;
    }

    void process_agent_trades(const std::vector<Trade>& trades,
                              Side agent_side,
                              std::size_t& fill_count) {
        for (const auto& t : trades) {
            if (agent_side == Side::Bid) {
                // We are the taker buying.
                inventory_ += static_cast<double>(t.qty);
                pnl_ -= static_cast<double>(t.price * t.qty);
            } else {
                // We are the taker selling.
                inventory_ -= static_cast<double>(t.qty);
                pnl_ += static_cast<double>(t.price * t.qty);
            }
            ++fill_count;
        }
    }

    /// Fill the pre-allocated observation buffer — zero allocation.
    void fill_obs() noexcept {
        const std::size_t N = cfg_.obs_depth;
        Price mid = compute_mid();
        double tick_d = static_cast<double>(cfg_.tick_size);

        // Zero the buffer first.
        for (auto& v : obs_buffer_) v = 0.0;

        // Bid levels: scan downward from best_bid.
        Price bb = book_.best_bid();
        if (bb != std::numeric_limits<Price>::min()) {
            std::size_t filled = 0;
            Price p = bb;
            while (filled < N && p >= cfg_.base_price && price_in_range(p)) {
                const auto& lvl = book_.level(Side::Bid, p);
                if (!lvl.queue.empty()) {
                    obs_buffer_[filled * 2]     =
                        static_cast<double>(p - mid) / tick_d;
                    obs_buffer_[filled * 2 + 1] =
                        static_cast<double>(lvl.queue.total_qty()) /
                        cfg_.max_volume;
                    ++filled;
                }
                p -= cfg_.tick_size;
            }
        }

        // Ask levels: scan upward from best_ask.
        std::size_t ask_offset = N * 2;
        Price ba = book_.best_ask();
        if (ba != std::numeric_limits<Price>::max()) {
            std::size_t filled = 0;
            Price p = ba;
            Price max_price = cfg_.base_price +
                              static_cast<Price>(cfg_.num_levels - 1) *
                              cfg_.tick_size;
            while (filled < N && p <= max_price && price_in_range(p)) {
                const auto& lvl = book_.level(Side::Ask, p);
                if (!lvl.queue.empty()) {
                    obs_buffer_[ask_offset + filled * 2] =
                        static_cast<double>(p - mid) / tick_d;
                    obs_buffer_[ask_offset + filled * 2 + 1] =
                        static_cast<double>(lvl.queue.total_qty()) /
                        cfg_.max_volume;
                    ++filled;
                }
                p += cfg_.tick_size;
            }
        }

        // Agent state.
        std::size_t state_offset = N * 4;
        obs_buffer_[state_offset]     = inventory_ / cfg_.max_inventory;
        obs_buffer_[state_offset + 1] = pnl_ / cfg_.max_pnl;
    }

    // ── Data ────────────────────────────────────────────────────────────────
    EnvConfig               cfg_;
    OrderBook<ENV_POOL_CAP> book_;
    HawkesSimulator         sim_;

    // Pre-allocated flat observation buffer — zero-copy to Python.
    std::vector<double>     obs_buffer_;

    // Generated ticks for the current episode.
    std::vector<Tick>       ticks_;
    std::size_t             tick_cursor_ = 0;

    // Agent state.
    double   inventory_    = 0.0;
    double   pnl_          = 0.0;
    double   prev_mtm_     = 0.0;
    std::size_t step_count_ = 0;

    // Agent order tracking.
    OrderId  next_agent_id_ = AGENT_ID_BASE;
    OrderId  agent_bid_id_  = 0;
    OrderId  agent_ask_id_  = 0;

    std::size_t episode_count_ = 0;
};

}  // namespace rcmm
