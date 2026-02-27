// ============================================================================
// risk-constrained-mm :: cpp/include/lob/order_queue.hpp
// ============================================================================
// An intrusive doubly-linked list representing all orders resting at a
// single price level.  Maintains FIFO (time-priority) ordering.
//
// Complexity guarantees:
//   push_back  — O(1)   (append to tail = lowest priority at this level)
//   remove     — O(1)   (given a pointer to the node)
//   pop_front  — O(1)   (dequeue highest priority order)
//
// No heap allocation — node storage is owned by the OrderPool.
// ============================================================================
#pragma once

#include "lob/order.hpp"

#include <cassert>
#include <cstddef>

namespace rcmm {

class OrderQueue {
public:
    // ── Construction ────────────────────────────────────────────────────────
    OrderQueue() noexcept = default;

    // Non-copyable (the queue does not own the nodes).
    OrderQueue(const OrderQueue&)            = delete;
    OrderQueue& operator=(const OrderQueue&) = delete;
    OrderQueue(OrderQueue&&)                 = delete;
    OrderQueue& operator=(OrderQueue&&)      = delete;

    // ── Modifiers ───────────────────────────────────────────────────────────
    /// Append `order` at the back (lowest time priority at this price).
    void push_back(Order* order) noexcept {
        assert(order != nullptr);
        assert(order->prev == nullptr && order->next == nullptr);

        order->prev = tail_;
        order->next = nullptr;

        if (tail_ != nullptr) {
            tail_->next = order;
        } else {
            head_ = order;   // list was empty
        }
        tail_ = order;
        ++count_;
        total_qty_ += order->leaves();
    }

    /// Remove an arbitrary node.  Caller guarantees `order` is in *this* queue.
    void remove(Order* order) noexcept {
        assert(order != nullptr);

        if (order->prev != nullptr) {
            order->prev->next = order->next;
        } else {
            head_ = order->next;          // removing head
        }

        if (order->next != nullptr) {
            order->next->prev = order->prev;
        } else {
            tail_ = order->prev;          // removing tail
        }

        total_qty_ -= order->leaves();
        order->prev = nullptr;
        order->next = nullptr;
        --count_;
    }

    /// Dequeue the highest-priority (oldest) order.  Returns nullptr if empty.
    [[nodiscard]] Order* pop_front() noexcept {
        if (head_ == nullptr) return nullptr;
        Order* front = head_;
        remove(front);
        return front;
    }

    // ── Queries ─────────────────────────────────────────────────────────────
    [[nodiscard]] Order*      front()     const noexcept { return head_; }
    [[nodiscard]] Order*      back()      const noexcept { return tail_; }
    [[nodiscard]] bool        empty()     const noexcept { return head_ == nullptr; }
    [[nodiscard]] std::size_t count()     const noexcept { return count_; }
    [[nodiscard]] Qty         total_qty() const noexcept { return total_qty_; }

    /// Adjust aggregate qty after a partial fill (caller updates Order::filled).
    void reduce_qty(Qty delta) noexcept {
        assert(delta > 0);
        total_qty_ -= delta;
    }

private:
    Order*      head_      = nullptr;
    Order*      tail_      = nullptr;
    std::size_t count_     = 0;
    Qty         total_qty_ = 0;       // sum of leaves() across all orders
};

}  // namespace rcmm
