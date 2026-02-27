// ============================================================================
// risk-constrained-mm :: tests/cpp/test_pool.cpp
// ============================================================================
// Unit tests for the OrderPool (zero-allocation memory pool).
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include "lob/pool.hpp"

using namespace rcmm;

// Use a small pool for fast, deterministic tests.
constexpr std::size_t TEST_POOL_CAP = 64;

TEST_CASE("OrderPool: initial state", "[pool]") {
    OrderPool<TEST_POOL_CAP> pool;

    REQUIRE(pool.capacity()   == TEST_POOL_CAP);
    REQUIRE(pool.live_count() == 0);
    REQUIRE(pool.free_count() == TEST_POOL_CAP);
}

TEST_CASE("OrderPool: allocate returns valid, zeroed orders", "[pool]") {
    OrderPool<TEST_POOL_CAP> pool;

    Order* o = pool.allocate();
    REQUIRE(o != nullptr);
    REQUIRE(o->id    == INVALID_ORDER_ID);
    REQUIRE(o->price == INVALID_PRICE);
    REQUIRE(o->qty   == 0);
    REQUIRE(o->prev  == nullptr);
    REQUIRE(o->next  == nullptr);

    REQUIRE(pool.live_count() == 1);
    REQUIRE(pool.free_count() == TEST_POOL_CAP - 1);
    REQUIRE(pool.owns(o));
}

TEST_CASE("OrderPool: deallocate returns slot to free list", "[pool]") {
    OrderPool<TEST_POOL_CAP> pool;

    Order* o = pool.allocate();
    REQUIRE(o != nullptr);

    pool.deallocate(o);
    REQUIRE(pool.live_count() == 0);
    REQUIRE(pool.free_count() == TEST_POOL_CAP);
}

TEST_CASE("OrderPool: exhaust entire pool then refill", "[pool]") {
    OrderPool<TEST_POOL_CAP> pool;

    Order* orders[TEST_POOL_CAP]{};

    // Allocate every slot.
    for (std::size_t i = 0; i < TEST_POOL_CAP; ++i) {
        orders[i] = pool.allocate();
        REQUIRE(orders[i] != nullptr);
    }
    REQUIRE(pool.live_count() == TEST_POOL_CAP);
    REQUIRE(pool.free_count() == 0);

    // Next alloc must return nullptr.
    REQUIRE(pool.allocate() == nullptr);

    // Free all, then re-allocate.
    for (std::size_t i = 0; i < TEST_POOL_CAP; ++i) {
        pool.deallocate(orders[i]);
    }
    REQUIRE(pool.live_count() == 0);
    REQUIRE(pool.free_count() == TEST_POOL_CAP);

    // Re-allocate all — should succeed.
    for (std::size_t i = 0; i < TEST_POOL_CAP; ++i) {
        REQUIRE(pool.allocate() != nullptr);
    }
    REQUIRE(pool.live_count() == TEST_POOL_CAP);
}

TEST_CASE("OrderPool: no duplicate pointers on sequential alloc", "[pool]") {
    OrderPool<TEST_POOL_CAP> pool;

    Order* a = pool.allocate();
    Order* b = pool.allocate();
    REQUIRE(a != b);
    REQUIRE(pool.owns(a));
    REQUIRE(pool.owns(b));
}

TEST_CASE("OrderPool: deallocated slot is reused", "[pool]") {
    OrderPool<TEST_POOL_CAP> pool;

    Order* a = pool.allocate();
    pool.deallocate(a);

    Order* b = pool.allocate();
    // The free-list is LIFO, so the same slot should be returned.
    REQUIRE(b == a);
}
