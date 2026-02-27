// ============================================================================
// risk-constrained-mm :: cpp/include/lob/order_book.hpp
// ============================================================================
// The central Limit Order Book.
//
// Architecture:
//
//   Price-level index
//   ─────────────────
//   Flat, pre-allocated arrays (one per side) indexed by a price offset from
//   a configurable base price.  O(1) lookup with perfect cache locality.
//
//   Best bid / ask tracking
//   ───────────────────────
//   Maintained as Price members, updated on every insert / cancel.
//   On cancel of the current best, we scan the level array to find the next
//   non-empty level — O(gap) worst-case but O(1) amortised.
//
//   Order lookup by ID
//   ──────────────────
//   A custom open-addressing hash map (OrderMap) pre-allocated at init
//   provides O(1) insert / find / erase with zero hot-path allocation.
//
// Public API:
//   place_order()  — match against resting orders, rest remainder (Phase 3).
//   add_order()    — allocate from pool, register in map, push to queue.
//   cancel_order() — lookup in map, unlink from queue, return to pool.
// ============================================================================
#pragma once

#include "lob/order.hpp"
#include "lob/order_map.hpp"
#include "lob/order_queue.hpp"
#include "lob/pool.hpp"
#include "lob/price_level.hpp"
#include "lob/types.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <memory>
#include <vector>

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

static_assert(std::is_trivially_copyable_v<Trade>,
              "Trade must be trivially copyable for flat storage");

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
    explicit OrderBook(BookConfig cfg = {})
        : cfg_(cfg),
          bid_levels_(new PriceLevel[cfg.num_levels]),
          ask_levels_(new PriceLevel[cfg.num_levels]),
          best_bid_(std::numeric_limits<Price>::min()),
          best_ask_(std::numeric_limits<Price>::max()),
          order_map_(PoolCap) {
        assert(cfg_.tick_size > 0);
        assert(cfg_.num_levels > 0);
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

    [[nodiscard]] const BookConfig& config()    const noexcept { return cfg_; }
    [[nodiscard]] const OrderPool<PoolCap>& pool() const noexcept { return pool_; }
    [[nodiscard]] const OrderMap& order_map()   const noexcept { return order_map_; }

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

    // ── Order Management ────────────────────────────────────────────────────

    /// Place a new limit order on the book.
    ///
    /// Returns a raw pointer to the order in the pool on success, or nullptr
    /// if the pool is exhausted, the ID is a duplicate, or the map is full.
    ///
    /// Hot-path: zero heap allocation — pool.allocate() + map.insert() +
    ///           queue.push_back() are all O(1) from pre-allocated storage.
    [[nodiscard]] Order* add_order(OrderId id, Side side, Price price, Qty qty,
                                   Timestamp ts = 0) noexcept {
        assert(id != INVALID_ORDER_ID);
        assert(qty > 0);
        assert(price >= cfg_.base_price);
        assert(price_to_index(price) < cfg_.num_levels);

        // 1. Acquire a slot from the memory pool.
        Order* order = pool_.allocate();
        if (order == nullptr) return nullptr;  // pool exhausted

        // 2. Populate order fields.
        order->id        = id;
        order->side      = side;
        order->type      = OrderType::Limit;
        order->status    = OrderStatus::New;
        order->price     = price;
        order->qty       = qty;
        order->filled    = 0;
        order->seq       = next_seq_++;
        order->timestamp = ts;

        // 3. Register in the O(1) ID lookup map.
        if (!order_map_.insert(id, order)) {
            // Duplicate ID or map full — roll back the pool allocation.
            pool_.deallocate(order);
            return nullptr;
        }

        // 4. Append to the price-level queue (FIFO / time priority).
        auto& lvl = level(side, price);
        lvl.queue.push_back(order);

        // 5. Update best bid / ask.
        if (side == Side::Bid && price > best_bid_) {
            best_bid_ = price;
        } else if (side == Side::Ask && price < best_ask_) {
            best_ask_ = price;
        }

        return order;
    }

    /// Cancel an existing order by its ID.
    ///
    /// Returns true if the order was found and removed, false if the ID is
    /// unknown (safe no-op).
    ///
    /// Hot-path: O(1) map lookup + O(1) intrusive-list unlink + O(1) pool
    ///           dealloc.  If the cancelled order was at the current best
    ///           price and the level is now empty, a linear scan finds the
    ///           next non-empty level.
    bool cancel_order(OrderId id) noexcept {
        // 1. O(1) lookup.
        Order* order = order_map_.find(id);
        if (order == nullptr) return false;

        Side  side  = order->side;
        Price price = order->price;

        // 2. O(1) unlink from the price-level queue.
        auto& lvl = level(side, price);
        lvl.queue.remove(order);

        // 3. Remove from the ID map.
        order_map_.erase(id);

        // 4. Return slot to the memory pool (calls order->reset()).
        pool_.deallocate(order);

        // 5. If this level is now empty and it was the best, rescan.
        if (lvl.queue.empty()) {
            if (side == Side::Bid && price == best_bid_) {
                best_bid_ = scan_best_bid(price);
            } else if (side == Side::Ask && price == best_ask_) {
                best_ask_ = scan_best_ask(price);
            }
        }

        return true;
    }

    /// Look up an active order by ID.  Returns nullptr if not found.
    [[nodiscard]] Order* find_order(OrderId id) const noexcept {
        return order_map_.find(id);
    }

    // ── Matching Engine (Phase 3) ──────────────────────────────────────────

    /// Place an order that may match against resting orders (crossing the
    /// spread).
    ///
    /// For Limit orders: matches against resting orders at favorable prices,
    /// then rests any remaining qty on the book.
    /// For Market orders: sweeps the opposite side until qty is zero or the
    /// book is empty.  Never rests on the book.
    ///
    /// Returns a vector of Trade events (one per fill).  The vector is
    /// reserve()'d to avoid repeated re-allocation.
    [[nodiscard]] std::vector<Trade> place_order(
        OrderId id, Price price, Qty qty, Side side, OrderType type,
        Timestamp ts = 0) {

        assert(id != INVALID_ORDER_ID);
        assert(qty > 0);
        assert(type == OrderType::Limit || type == OrderType::Market);
        // Limit orders must target a valid book price (for potential resting).
        assert(type != OrderType::Limit ||
               (price >= cfg_.base_price &&
                price_to_index(price) < cfg_.num_levels));

        std::vector<Trade> trades;
        trades.reserve(64);

        Qty remaining = qty;

        if (side == Side::Bid) {
            // Aggressive buy: match against asks (lowest first = price prio).
            while (remaining > 0 &&
                   best_ask_ < std::numeric_limits<Price>::max()) {
                if (type == OrderType::Limit && best_ask_ > price) break;

                auto& lvl = level(Side::Ask, best_ask_);
                while (remaining > 0 && !lvl.queue.empty()) {
                    Order* maker = lvl.queue.front();
                    Qty fill = std::min(remaining, maker->leaves());

                    trades.push_back(Trade{
                        maker->id, id, best_ask_, fill, side, ts});

                    remaining -= fill;
                    maker->filled += fill;
                    lvl.queue.reduce_qty(fill);

                    if (maker->leaves() == 0) {
                        maker->status = OrderStatus::Filled;
                        lvl.queue.remove(maker);
                        order_map_.erase(maker->id);
                        pool_.deallocate(maker);
                    } else {
                        maker->status = OrderStatus::PartialFill;
                    }
                }

                if (lvl.queue.empty()) {
                    best_ask_ = scan_best_ask(best_ask_);
                }
            }
        } else {
            // Aggressive sell: match against bids (highest first).
            while (remaining > 0 &&
                   best_bid_ > std::numeric_limits<Price>::min()) {
                if (type == OrderType::Limit && best_bid_ < price) break;

                auto& lvl = level(Side::Bid, best_bid_);
                while (remaining > 0 && !lvl.queue.empty()) {
                    Order* maker = lvl.queue.front();
                    Qty fill = std::min(remaining, maker->leaves());

                    trades.push_back(Trade{
                        maker->id, id, best_bid_, fill, side, ts});

                    remaining -= fill;
                    maker->filled += fill;
                    lvl.queue.reduce_qty(fill);

                    if (maker->leaves() == 0) {
                        maker->status = OrderStatus::Filled;
                        lvl.queue.remove(maker);
                        order_map_.erase(maker->id);
                        pool_.deallocate(maker);
                    } else {
                        maker->status = OrderStatus::PartialFill;
                    }
                }

                if (lvl.queue.empty()) {
                    best_bid_ = scan_best_bid(best_bid_);
                }
            }
        }

        // Limit orders: rest unfilled remainder on the book.
        if (type == OrderType::Limit && remaining > 0) {
            (void)add_order(id, side, price, remaining, ts);
        }

        return trades;
    }

protected:
    // ── Helpers ─────────────────────────────────────────────────────────────
    [[nodiscard]] std::size_t price_to_index(Price p) const noexcept {
        auto idx = static_cast<std::size_t>((p - cfg_.base_price) / cfg_.tick_size);
        assert(idx < cfg_.num_levels);
        return idx;
    }

    /// Scan downward from `from` (exclusive) to find the next non-empty bid.
    /// Returns the sentinel min if no bids remain.
    [[nodiscard]] Price scan_best_bid(Price from) const noexcept {
        std::size_t idx = price_to_index(from);
        while (idx > 0) {
            --idx;
            if (!bid_levels_[idx].queue.empty()) {
                return bid_levels_[idx].price;
            }
        }
        return std::numeric_limits<Price>::min();
    }

    /// Scan upward from `from` (exclusive) to find the next non-empty ask.
    /// Returns the sentinel max if no asks remain.
    [[nodiscard]] Price scan_best_ask(Price from) const noexcept {
        std::size_t idx = price_to_index(from);
        for (std::size_t i = idx + 1; i < cfg_.num_levels; ++i) {
            if (!ask_levels_[i].queue.empty()) {
                return ask_levels_[i].price;
            }
        }
        return std::numeric_limits<Price>::max();
    }

    // ── Data ────────────────────────────────────────────────────────────────
    BookConfig cfg_;

    // Flat arrays of price levels — one per side.
    // Heap-allocated once at construction via unique_ptr<PriceLevel[]>,
    // giving contiguous, cache-friendly storage without blowing the stack.
    std::unique_ptr<PriceLevel[]> bid_levels_;
    std::unique_ptr<PriceLevel[]> ask_levels_;

    Price best_bid_;
    Price best_ask_;

    OrderPool<PoolCap> pool_;
    OrderMap            order_map_;

    Sequence next_seq_ = 1;   // monotonic sequence counter
};

}  // namespace rcmm
