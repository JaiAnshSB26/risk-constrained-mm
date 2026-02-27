// ============================================================================
// risk-constrained-mm :: cpp/include/lob/order_book.hpp
// ============================================================================
// The central Limit Order Book.
//
// Architecture (Phase 1 skeleton — matching logic added in Phase 3):
//
//   Price-level index
//   ─────────────────
//   We use a flat, pre-allocated array indexed by a *price offset* from a
//   configurable base price.  For crypto pairs with bounded tick ranges this
//   gives O(1) price-level lookup with perfect cache locality.
//
//   For instruments where the price range is very wide  we will swap in
//   a robin-hood hash map in Phase 2 (behind a compile-time policy).
//
//   Best bid / ask tracking
//   ───────────────────────
//   Maintained as simple Price members updated on every insert / cancel.
//
//   Order lookup by ID
//   ──────────────────
//   A flat array indexed by (id % capacity) or a robin-hood map provides
//   O(1) cancel.  Implemented in Phase 2 with the full add/cancel API.
//
// This header defines the public interface that the matching engine and
// the Python bindings will program against.
// ============================================================================
#pragma once

#include "lob/order.hpp"
#include "lob/order_queue.hpp"
#include "lob/pool.hpp"
#include "lob/price_level.hpp"
#include "lob/types.hpp"

#include <array>
#include <cstddef>
#include <limits>

namespace rcmm {

/// Execution report produced when two orders match (Phase 3).
struct Trade {
    OrderId   maker_id   = INVALID_ORDER_ID;
    OrderId   taker_id   = INVALID_ORDER_ID;
    Price     price      = INVALID_PRICE;
    Qty       qty        = 0;
    Side      taker_side = Side::Bid;
    Timestamp timestamp  = 0;
};

/// Configuration for an OrderBook instance.
struct BookConfig {
    Price       tick_size   = 1;          // smallest price increment
    Price       base_price  = 0;          // low end of the price array
    std::size_t num_levels  = MAX_PRICE_LEVELS;  // slots in the flat array
    std::size_t pool_cap    = MAX_ORDERS;         // order pool capacity
};

/// The Limit Order Book.
/// Template parameter `PoolCap` sizes the embedded OrderPool.
template <std::size_t PoolCap = MAX_ORDERS>
class OrderBook {
public:
    // ── Construction ────────────────────────────────────────────────────────
    explicit OrderBook(BookConfig cfg = {}) noexcept
        : cfg_(cfg),
          best_bid_(std::numeric_limits<Price>::min()),
          best_ask_(std::numeric_limits<Price>::max()) {
        // Stamp each level with its price.
        for (std::size_t i = 0; i < cfg_.num_levels; ++i) {
            bid_levels_[i].price = cfg_.base_price + static_cast<Price>(i) * cfg_.tick_size;
            ask_levels_[i].price = cfg_.base_price + static_cast<Price>(i) * cfg_.tick_size;
        }
    }

    // Non-copyable, non-movable.
    OrderBook(const OrderBook&)            = delete;
    OrderBook& operator=(const OrderBook&) = delete;
    OrderBook(OrderBook&&)                 = delete;
    OrderBook& operator=(OrderBook&&)      = delete;

    // ── Accessors (always O(1)) ─────────────────────────────────────────────
    [[nodiscard]] Price best_bid() const noexcept { return best_bid_; }
    [[nodiscard]] Price best_ask() const noexcept { return best_ask_; }
    [[nodiscard]] Price spread()   const noexcept { return best_ask_ - best_bid_; }

    [[nodiscard]] const BookConfig& config() const noexcept { return cfg_; }

    [[nodiscard]] const OrderPool<PoolCap>& pool() const noexcept { return pool_; }

    // ── Price-level access ──────────────────────────────────────────────────
    /// Returns the PriceLevel for the given price on the specified side.
    /// Precondition: price_to_index(price) < cfg_.num_levels.
    [[nodiscard]] PriceLevel& level(Side side, Price price) noexcept {
        auto idx = price_to_index(price);
        return (side == Side::Bid) ? bid_levels_[idx] : ask_levels_[idx];
    }

    [[nodiscard]] const PriceLevel& level(Side side, Price price) const noexcept {
        auto idx = price_to_index(price);
        return (side == Side::Bid) ? bid_levels_[idx] : ask_levels_[idx];
    }

    // ── Phase 2/3 stubs ─────────────────────────────────────────────────────
    // These will be implemented in Phases 2 & 3.
    // Order* add_order(Side, Price, Qty, Timestamp);
    // bool   cancel_order(OrderId);
    // std::span<Trade> match();

protected:
    // ── Helpers ─────────────────────────────────────────────────────────────
    [[nodiscard]] std::size_t price_to_index(Price p) const noexcept {
        auto idx = static_cast<std::size_t>((p - cfg_.base_price) / cfg_.tick_size);
        assert(idx < cfg_.num_levels);
        return idx;
    }

    // ── Data ────────────────────────────────────────────────────────────────
    BookConfig cfg_;

    // Flat arrays of price levels — one per side.
    // Heap-allocated via std::vector to allow runtime cfg_.num_levels sizing
    // while still providing contiguous, cache-friendly storage.
    // For Phase 1 we use a fixed std::array sized to MAX_PRICE_LEVELS.
    std::array<PriceLevel, MAX_PRICE_LEVELS> bid_levels_{};
    std::array<PriceLevel, MAX_PRICE_LEVELS> ask_levels_{};

    Price best_bid_;
    Price best_ask_;

    OrderPool<PoolCap> pool_;

    Sequence next_seq_ = 1;   // monotonic sequence counter
};

}  // namespace rcmm
