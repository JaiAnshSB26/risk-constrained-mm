// ============================================================================
// risk-constrained-mm :: cpp/include/data/replay_engine.hpp
// ============================================================================
// The ReplayEngine feeds parsed Tick data into a live OrderBook, applying
// each event through the existing add_order / cancel_order / place_order
// API.  It also exposes a top-of-book (TOB) snapshot at any point.
//
// Tick action mapping:
//   Add    -> book.add_order()
//   Cancel -> book.cancel_order()
//   Modify -> book.cancel_order() + book.add_order()  (atomic replace)
//   Trade  -> book.place_order(Market)                 (aggressive cross)
//
// ============================================================================
#pragma once

#include "data/tick.hpp"
#include "lob/order_book.hpp"

#include <cstddef>
#include <limits>
#include <vector>

namespace rcmm {

/// Snapshot of the top of the order book at a given timestamp.
struct TopOfBook {
    Price     best_bid  = std::numeric_limits<Price>::min();
    Price     best_ask  = std::numeric_limits<Price>::max();
    Timestamp timestamp = 0;
};

/// Aggregate result of a batch replay.
struct ReplayResult {
    std::vector<Trade> trades;            // all trades generated
    std::size_t        ticks_processed = 0;
    std::size_t        errors          = 0;
};

/// Replays historical tick data onto a live OrderBook.
template <std::size_t PoolCap = MAX_ORDERS>
class ReplayEngine {
public:
    explicit ReplayEngine(OrderBook<PoolCap>& book) noexcept
        : book_(book) {}

    // Non-copyable, non-movable.
    ReplayEngine(const ReplayEngine&)            = delete;
    ReplayEngine& operator=(const ReplayEngine&) = delete;
    ReplayEngine(ReplayEngine&&)                 = delete;
    ReplayEngine& operator=(ReplayEngine&&)      = delete;

    // ── Batch replay ────────────────────────────────────────────────────────

    /// Replay a full sequence of ticks onto the OrderBook.
    /// Returns an aggregate result with all trades, processed count, and
    /// error count.
    [[nodiscard]] ReplayResult replay(const std::vector<Tick>& ticks) {
        ReplayResult result;
        result.trades.reserve(ticks.size());

        for (const auto& tick : ticks) {
            current_ts_ = tick.timestamp;
            bool ok = false;

            switch (tick.action) {
            case TickAction::Add:
                ok = process_add(tick);
                break;
            case TickAction::Cancel:
                ok = process_cancel(tick);
                break;
            case TickAction::Modify:
                ok = process_modify(tick);
                break;
            case TickAction::Trade: {
                auto trades = process_trade(tick);
                result.trades.insert(result.trades.end(),
                                     trades.begin(), trades.end());
                ok = true;  // trade may produce 0 fills (empty book)
                break;
            }
            }

            if (ok) {
                ++result.ticks_processed;
            } else {
                ++result.errors;
            }
        }

        return result;
    }

    // ── Single-tick replay ──────────────────────────────────────────────────

    /// Replay one tick.  Returns any trades produced (empty for non-trade
    /// actions).  Useful for step-by-step RL environment interaction.
    [[nodiscard]] std::vector<Trade> replay_one(const Tick& tick) {
        current_ts_ = tick.timestamp;
        std::vector<Trade> trades;

        switch (tick.action) {
        case TickAction::Add:
            (void)process_add(tick);
            break;
        case TickAction::Cancel:
            (void)process_cancel(tick);
            break;
        case TickAction::Modify:
            (void)process_modify(tick);
            break;
        case TickAction::Trade:
            trades = process_trade(tick);
            break;
        }

        return trades;
    }

    // ── State queries ───────────────────────────────────────────────────────

    /// Current top-of-book snapshot.
    [[nodiscard]] TopOfBook top_of_book() const noexcept {
        return {book_.best_bid(), book_.best_ask(), current_ts_};
    }

    /// Most recent timestamp seen during replay.
    [[nodiscard]] Timestamp current_timestamp() const noexcept {
        return current_ts_;
    }

    /// Const access to the underlying book.
    [[nodiscard]] const OrderBook<PoolCap>& book() const noexcept {
        return book_;
    }

private:
    // ── Action handlers ─────────────────────────────────────────────────────

    bool process_add(const Tick& tick) noexcept {
        return book_.add_order(
            tick.order_id, tick.side, tick.price,
            tick.qty, tick.timestamp) != nullptr;
    }

    bool process_cancel(const Tick& tick) noexcept {
        return book_.cancel_order(tick.order_id);
    }

    bool process_modify(const Tick& tick) noexcept {
        // Modify = cancel old + add with new parameters.
        // Cancel may fail if the order was already consumed — that is OK.
        book_.cancel_order(tick.order_id);
        return book_.add_order(
            tick.order_id, tick.side, tick.price,
            tick.qty, tick.timestamp) != nullptr;
    }

    [[nodiscard]] std::vector<Trade> process_trade(const Tick& tick) {
        return book_.place_order(
            tick.order_id, tick.price, tick.qty, tick.side,
            OrderType::Market, tick.timestamp);
    }

    // ── Data ────────────────────────────────────────────────────────────────
    OrderBook<PoolCap>& book_;
    Timestamp           current_ts_ = 0;
};

}  // namespace rcmm
