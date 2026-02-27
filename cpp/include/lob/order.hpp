// ============================================================================
// risk-constrained-mm :: cpp/include/lob/order.hpp
// ============================================================================
// The Order struct lives inside a pre-allocated pool (see pool.hpp).
// It carries intrusive doubly-linked-list pointers so that OrderQueue
// (a single price level) can chain Orders with zero heap allocation.
//
// Memory layout is cache-line–aware: hot fields (price, qty, pointers)
// are packed into the first 64 bytes.
// ============================================================================
#pragma once

#include "lob/types.hpp"

namespace rcmm {

struct Order {
    // ── Identity ────────────────────────────────────────────────────────────
    OrderId    id        = INVALID_ORDER_ID;
    Symbol     symbol    = 0;
    Side       side      = Side::Bid;
    OrderType  type      = OrderType::Limit;
    OrderStatus status   = OrderStatus::New;

    // ── Economics ───────────────────────────────────────────────────────────
    Price      price     = INVALID_PRICE;
    Qty        qty       = 0;             // remaining quantity
    Qty        filled    = 0;             // cumulative filled

    // ── Time priority ───────────────────────────────────────────────────────
    Sequence   seq       = 0;             // set by the engine on arrival
    Timestamp  timestamp = 0;             // exchange timestamp (for replay)

    // ── Intrusive doubly-linked-list pointers ───────────────────────────────
    // These are managed exclusively by OrderQueue.  nullptr ⇒ not in a queue.
    Order*     prev      = nullptr;
    Order*     next      = nullptr;

    // ── Helpers ─────────────────────────────────────────────────────────────
    [[nodiscard]] constexpr bool is_active() const noexcept {
        return status == OrderStatus::New || status == OrderStatus::PartialFill;
    }

    [[nodiscard]] constexpr Qty leaves() const noexcept {
        return qty - filled;
    }

    /// Reset all fields so the Pool can reuse this slot.
    constexpr void reset() noexcept {
        id        = INVALID_ORDER_ID;
        symbol    = 0;
        side      = Side::Bid;
        type      = OrderType::Limit;
        status    = OrderStatus::New;
        price     = INVALID_PRICE;
        qty       = 0;
        filled    = 0;
        seq       = 0;
        timestamp = 0;
        prev      = nullptr;
        next      = nullptr;
    }
};

// Compile-time sanity: Order must be trivially copyable (no hidden vtable,
// no std::string, etc.) so it can live safely in a flat pool.
static_assert(std::is_trivially_copyable_v<Order>,
              "Order must be trivially copyable for pool allocation");

}  // namespace rcmm
