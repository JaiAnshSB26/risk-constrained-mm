// ============================================================================
// risk-constrained-mm :: cpp/include/lob/pool.hpp
// ============================================================================
// Fixed-capacity, pre-allocated object pool for Order structs.
//
// Design:
//   • A single contiguous std::array<Order, Capacity> is allocated at
//     compile time (or wrapped in a vector sized once at init).
//   • A free-list (singly-linked through a reinterpret of Order::next)
//     provides O(1) alloc / dealloc with zero heap traffic on the hot path.
//   • No new, no malloc, no exceptions — ever.
//
// This is the foundation of the "zero-allocation matching engine" guarantee.
// ============================================================================
#pragma once

#include "lob/order.hpp"

#include <array>
#include <cassert>
#include <cstddef>

namespace rcmm {

/// Compile-time–sized pool.  `Capacity` defaults to MAX_ORDERS.
template <std::size_t Capacity = MAX_ORDERS>
class OrderPool {
public:
    // ── Construction ────────────────────────────────────────────────────────
    OrderPool() noexcept { build_free_list(); }

    // Non-copyable, non-movable (the pool owns the storage).
    OrderPool(const OrderPool&)            = delete;
    OrderPool& operator=(const OrderPool&) = delete;
    OrderPool(OrderPool&&)                 = delete;
    OrderPool& operator=(OrderPool&&)      = delete;

    // ── Allocate ────────────────────────────────────────────────────────────
    /// Acquire a zeroed Order slot from the free list.  Returns nullptr if
    /// the pool is exhausted (caller must check — we never throw).
    [[nodiscard]] Order* allocate() noexcept {
        if (free_head_ == nullptr) {
            return nullptr;           // pool exhausted
        }
        Order* slot  = free_head_;
        free_head_   = free_head_->next;   // pop from free list
        slot->reset();                     // zero all fields
        ++live_count_;
        return slot;
    }

    // ── Deallocate ──────────────────────────────────────────────────────────
    /// Return an order slot to the free list.  O(1).
    /// Precondition: `order` was obtained from *this* pool.
    void deallocate(Order* order) noexcept {
        assert(order != nullptr);
        assert(owns(order));               // debug-mode bounds check
        order->reset();
        order->next = free_head_;          // push onto free list
        free_head_  = order;
        --live_count_;
    }

    // ── Queries ─────────────────────────────────────────────────────────────
    [[nodiscard]] constexpr std::size_t capacity()   const noexcept { return Capacity; }
    [[nodiscard]]           std::size_t live_count() const noexcept { return live_count_; }
    [[nodiscard]]           std::size_t free_count() const noexcept { return Capacity - live_count_; }

    /// True if `order` points into our storage array.
    [[nodiscard]] bool owns(const Order* order) const noexcept {
        const Order* begin = storage_.data();
        const Order* end   = storage_.data() + Capacity;
        return order >= begin && order < end;
    }

private:
    void build_free_list() noexcept {
        // Chain every slot through Order::next.  The last slot's next is nullptr.
        for (std::size_t i = 0; i + 1 < Capacity; ++i) {
            storage_[i].next = &storage_[i + 1];
        }
        storage_[Capacity - 1].next = nullptr;
        free_head_ = &storage_[0];
    }

    // ── Data ────────────────────────────────────────────────────────────────
    std::array<Order, Capacity> storage_{};
    Order*                      free_head_   = nullptr;
    std::size_t                 live_count_  = 0;
};

}  // namespace rcmm
