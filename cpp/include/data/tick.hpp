// ============================================================================
// risk-constrained-mm :: cpp/include/data/tick.hpp
// ============================================================================
// The Tick struct represents a single row of historical market data (L3).
//
// Fields mirror the Tardis.dev / Binance normalised feed:
//   timestamp, order_id, side, price, qty, action (add/cancel/modify/trade).
//
// Trivially copyable — safe for flat arrays and memory-mapped replay.
// ============================================================================
#pragma once

#include "lob/types.hpp"

#include <cstdint>
#include <type_traits>

namespace rcmm {

// ── Tick action ─────────────────────────────────────────────────────────────
enum class TickAction : std::uint8_t {
    Add    = 0,
    Cancel = 1,
    Modify = 2,
    Trade  = 3,
};

// ── Tick ────────────────────────────────────────────────────────────────────
struct Tick {
    Timestamp  timestamp = 0;               // nanoseconds since epoch
    OrderId    order_id  = INVALID_ORDER_ID;
    Price      price     = INVALID_PRICE;   // in integer tick units
    Qty        qty       = 0;               // in integer lot units
    Side       side      = Side::Bid;
    TickAction action    = TickAction::Add;
    bool       is_trade  = false;           // convenience: action == Trade
};

static_assert(std::is_trivially_copyable_v<Tick>,
              "Tick must be trivially copyable for flat storage");

}  // namespace rcmm
