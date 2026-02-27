// ============================================================================
// risk-constrained-mm :: cpp/include/lob/types.hpp
// ============================================================================
// Fundamental types used across the entire LOB engine.
//
// Design principles:
//   • Fixed-width integers for deterministic cross-platform behaviour.
//   • Price represented as int64_t in tick units (no floating-point on the
//     critical path — avoids rounding drift and is faster on modern CPUs).
//   • Every struct is trivially copyable / memcpy-safe.
// ============================================================================
#pragma once

#include <cstddef>
#include <cstdint>

namespace rcmm {

// ── Identifiers ─────────────────────────────────────────────────────────────
using OrderId  = std::uint64_t;
using Sequence = std::uint64_t;  // monotonic insertion counter (time priority)
using Symbol   = std::uint32_t;  // compact instrument id

// ── Price / Qty ─────────────────────────────────────────────────────────────
// Price is stored in *tick units* (integer).  For BTC/USDT with a tick size
// of 0.01 USDT, a price of 65432.10 is stored as 6'543'210.
using Price    = std::int64_t;
using Qty      = std::int64_t;   // signed to allow delta calculations
using Timestamp = std::int64_t;  // nanoseconds since epoch (matches Tardis)

// ── Enumerations ────────────────────────────────────────────────────────────
enum class Side : std::uint8_t {
    Bid = 0,
    Ask = 1,
};

enum class OrderType : std::uint8_t {
    Limit  = 0,
    Market = 1,
    Cancel = 2,
};

enum class OrderStatus : std::uint8_t {
    New           = 0,
    PartialFill   = 1,
    Filled        = 2,
    Cancelled     = 3,
    Rejected      = 4,
};

// ── Constants ───────────────────────────────────────────────────────────────
inline constexpr OrderId   INVALID_ORDER_ID  = 0;
inline constexpr Price     INVALID_PRICE     = 0;
inline constexpr std::size_t MAX_PRICE_LEVELS = 1 << 16;  // 65 536 levels
inline constexpr std::size_t MAX_ORDERS       = 1 << 20;  // ~1 million orders

}  // namespace rcmm
