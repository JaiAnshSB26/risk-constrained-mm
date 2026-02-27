// ============================================================================
// risk-constrained-mm :: cpp/include/lob/order_map.hpp
// ============================================================================
// Pre-allocated, open-addressing hash map:  OrderId  →  Order*
//
// Design:
//   • Power-of-2 table capacity with bitmask for fast modulo.
//   • Linear probing for cache-friendly sequential access on collisions.
//   • Backward-shift deletion (no tombstones) — keeps probe chains compact
//     and avoids the "graveyard" degradation problem.
//   • splitmix64 finalizer for hash distribution.
//   • Single contiguous heap allocation at construction; absolutely zero
//     heap traffic on insert / find / erase (the hot-path guarantee).
//
// Load factor:  Table is sized to ≥ 2× max_elements (next power of 2).
//               We hard-cap at 50 % occupancy for worst-case O(1) probing.
//
// Complexity (amortised):
//   insert  — O(1)
//   find    — O(1)
//   erase   — O(1)
// ============================================================================
#pragma once

#include "lob/order.hpp"
#include "lob/types.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace rcmm {

class OrderMap {
public:
    // ── Construction ────────────────────────────────────────────────────────
    /// Build a table that can hold at least `max_elements` entries.
    /// Actual capacity = next power of 2 ≥ 2 * max_elements.
    explicit OrderMap(std::size_t max_elements)
        : capacity_(next_pow2(max_elements < 4 ? 8 : max_elements * 2)),
          mask_(capacity_ - 1),
          slots_(std::make_unique<Slot[]>(capacity_)) {}

    // Non-copyable, non-movable (owns the slot storage).
    OrderMap(const OrderMap&)            = delete;
    OrderMap& operator=(const OrderMap&) = delete;
    OrderMap(OrderMap&&)                 = delete;
    OrderMap& operator=(OrderMap&&)      = delete;

    // ── Modifiers ───────────────────────────────────────────────────────────

    /// Insert (key → value).  Returns true on success.
    /// Returns false if key already exists OR table is at max load.
    /// Precondition: key != INVALID_ORDER_ID, value != nullptr.
    [[nodiscard]] bool insert(OrderId key, Order* value) noexcept {
        assert(key != INVALID_ORDER_ID);
        assert(value != nullptr);
        if (count_ >= capacity_ / 2) return false;  // enforce ≤ 50 % load

        std::size_t idx = slot_for(key);
        for (std::size_t i = 0; i < capacity_; ++i) {
            std::size_t pos = (idx + i) & mask_;
            if (!slots_[pos].occupied()) {
                slots_[pos] = {key, value};
                ++count_;
                return true;
            }
            if (slots_[pos].key == key) {
                return false;  // duplicate key
            }
        }
        return false;  // unreachable if load factor is respected
    }

    /// Look up an order by ID.  Returns nullptr if not found.
    [[nodiscard]] Order* find(OrderId key) const noexcept {
        assert(key != INVALID_ORDER_ID);

        std::size_t idx = slot_for(key);
        for (std::size_t i = 0; i < capacity_; ++i) {
            std::size_t pos = (idx + i) & mask_;
            if (!slots_[pos].occupied()) return nullptr;  // empty → miss
            if (slots_[pos].key == key)  return slots_[pos].value;
        }
        return nullptr;
    }

    /// Remove entry by key.  Returns true if found and removed, false if
    /// the key was not in the map.
    bool erase(OrderId key) noexcept {
        assert(key != INVALID_ORDER_ID);

        // 1. Locate the slot.
        std::size_t idx = slot_for(key);
        std::size_t pos = capacity_;  // sentinel: "not found"
        for (std::size_t i = 0; i < capacity_; ++i) {
            std::size_t p = (idx + i) & mask_;
            if (!slots_[p].occupied()) return false;  // miss
            if (slots_[p].key == key) { pos = p; break; }
        }
        if (pos == capacity_) return false;

        // 2. Backward-shift deletion — keeps probe chains intact, no tombstones.
        backward_shift(pos);
        --count_;
        return true;
    }

    // ── Queries ─────────────────────────────────────────────────────────────
    [[nodiscard]] std::size_t size()     const noexcept { return count_; }
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] bool        empty()    const noexcept { return count_ == 0; }

private:
    // ── Slot ────────────────────────────────────────────────────────────────
    struct Slot {
        OrderId key   = INVALID_ORDER_ID;
        Order*  value = nullptr;

        [[nodiscard]] constexpr bool occupied() const noexcept {
            return key != INVALID_ORDER_ID;
        }
    };

    // ── Hash function ───────────────────────────────────────────────────────
    /// splitmix64 finalizer — excellent avalanche on 64-bit integers.
    static constexpr std::uint64_t hash(OrderId key) noexcept {
        auto x = static_cast<std::uint64_t>(key);
        x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
        x = x ^ (x >> 31);
        return x;
    }

    /// Map a key to its ideal slot index.
    [[nodiscard]] std::size_t slot_for(OrderId key) const noexcept {
        return static_cast<std::size_t>(hash(key)) & mask_;
    }

    // ── Power-of-2 rounding ─────────────────────────────────────────────────
    static constexpr std::size_t next_pow2(std::size_t n) noexcept {
        if (n == 0) return 1;
        --n;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        n |= n >> 32;
        return n + 1;
    }

    // ── Backward-shift deletion ─────────────────────────────────────────────
    /// After logically removing the entry at `gap`, shift subsequent entries
    /// backwards to keep every key reachable from its ideal slot via an
    /// unbroken chain of occupied slots.
    void backward_shift(std::size_t gap) noexcept {
        std::size_t j = gap;
        for (;;) {
            j = (j + 1) & mask_;
            if (!slots_[j].occupied()) break;

            std::size_t k = slot_for(slots_[j].key);

            // Shift j → gap  iff  k is NOT in the cyclic range (gap, j].
            // If k IS in (gap, j], the entry probed into that range naturally
            // and must stay put.
            if (!in_cyclic_range(k, gap, j)) {
                slots_[gap] = slots_[j];
                gap = j;
            }
        }
        slots_[gap] = {INVALID_ORDER_ID, nullptr};
    }

    /// Is `x` in the circular half-open range (a, b]  (mod capacity)?
    [[nodiscard]] bool in_cyclic_range(std::size_t x,
                                       std::size_t a,
                                       std::size_t b) const noexcept {
        if (a < b) return (a < x) && (x <= b);
        if (a > b) return (a < x) || (x <= b);   // wraps around
        return false;  // a == b ⇒ empty range
    }

    // ── Data ────────────────────────────────────────────────────────────────
    std::size_t                 capacity_;
    std::size_t                 mask_;
    std::unique_ptr<Slot[]>     slots_;
    std::size_t                 count_ = 0;
};

}  // namespace rcmm
