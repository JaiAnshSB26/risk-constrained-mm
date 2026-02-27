// ============================================================================
// risk-constrained-mm :: cpp/include/lob/price_level.hpp
// ============================================================================
// Thin wrapper binding a Price to its OrderQueue.
// Used as the value type in the OrderBook's price-level index.
// ============================================================================
#pragma once

#include "lob/order_queue.hpp"
#include "lob/types.hpp"

namespace rcmm {

struct PriceLevel {
    Price      price = INVALID_PRICE;
    OrderQueue queue;

    // A level is "live" if it has at least one resting order.
    [[nodiscard]] bool empty() const noexcept { return queue.empty(); }
};

}  // namespace rcmm
