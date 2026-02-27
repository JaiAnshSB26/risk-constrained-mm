// ============================================================================
// risk-constrained-mm :: tests/cpp/test_order_map.cpp
// ============================================================================
// Unit tests for the pre-allocated open-addressing OrderMap.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include "lob/order_map.hpp"
#include "lob/pool.hpp"

using namespace rcmm;

// Small pool so we have real Order* pointers to store in the map.
constexpr std::size_t TEST_POOL_CAP = 64;

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST_CASE("OrderMap: initial state is empty", "[map]") {
    OrderMap map(32);

    REQUIRE(map.size()  == 0);
    REQUIRE(map.empty());
    // Capacity must be a power of 2, ≥ 2 × max_elements.
    REQUIRE(map.capacity() >= 64);
    REQUIRE((map.capacity() & (map.capacity() - 1)) == 0);  // power of 2
}

TEST_CASE("OrderMap: small capacity is clamped to minimum", "[map]") {
    OrderMap map(1);
    // Even for tiny max_elements, capacity should be a sane power of 2.
    REQUIRE(map.capacity() >= 8);
}

// ---------------------------------------------------------------------------
// Insert & Find
// ---------------------------------------------------------------------------

TEST_CASE("OrderMap: insert and find single entry", "[map]") {
    OrderPool<TEST_POOL_CAP> pool;
    OrderMap map(TEST_POOL_CAP);

    Order* o = pool.allocate();
    REQUIRE(o != nullptr);
    o->id = 42;

    REQUIRE(map.insert(42, o));
    REQUIRE(map.size() == 1);

    Order* found = map.find(42);
    REQUIRE(found == o);
    REQUIRE(found->id == 42);
}

TEST_CASE("OrderMap: find returns nullptr for missing key", "[map]") {
    OrderMap map(32);

    REQUIRE(map.find(999) == nullptr);
}

TEST_CASE("OrderMap: insert duplicate key is rejected", "[map]") {
    OrderPool<TEST_POOL_CAP> pool;
    OrderMap map(TEST_POOL_CAP);

    Order* a = pool.allocate();
    a->id = 1;
    Order* b = pool.allocate();
    b->id = 1;

    REQUIRE(map.insert(1, a));
    REQUIRE_FALSE(map.insert(1, b));  // duplicate
    REQUIRE(map.size() == 1);
    REQUIRE(map.find(1) == a);        // original still there
}

// ---------------------------------------------------------------------------
// Erase
// ---------------------------------------------------------------------------

TEST_CASE("OrderMap: erase existing entry", "[map]") {
    OrderPool<TEST_POOL_CAP> pool;
    OrderMap map(TEST_POOL_CAP);

    Order* o = pool.allocate();
    o->id = 7;
    REQUIRE(map.insert(7, o));

    REQUIRE(map.erase(7));
    REQUIRE(map.size() == 0);
    REQUIRE(map.find(7) == nullptr);
}

TEST_CASE("OrderMap: erase non-existent key returns false", "[map]") {
    OrderMap map(32);

    REQUIRE_FALSE(map.erase(123));
}

TEST_CASE("OrderMap: insert-erase-reinsert cycle", "[map]") {
    OrderPool<TEST_POOL_CAP> pool;
    OrderMap map(TEST_POOL_CAP);

    Order* a = pool.allocate();
    a->id = 10;
    Order* b = pool.allocate();
    b->id = 10;

    // Insert, erase, then re-insert the same key with a different pointer.
    REQUIRE(map.insert(10, a));
    REQUIRE(map.erase(10));
    REQUIRE(map.insert(10, b));
    REQUIRE(map.find(10) == b);
    REQUIRE(map.size() == 1);
}

// ---------------------------------------------------------------------------
// Stress: many insertions, lookups, erasures
// ---------------------------------------------------------------------------

TEST_CASE("OrderMap: bulk insert, find, and erase", "[map]") {
    constexpr std::size_t N = 50;
    OrderPool<TEST_POOL_CAP> pool;
    OrderMap map(TEST_POOL_CAP);

    Order* orders[N]{};

    // Insert N orders.
    for (std::size_t i = 0; i < N; ++i) {
        orders[i] = pool.allocate();
        REQUIRE(orders[i] != nullptr);
        orders[i]->id = static_cast<OrderId>(i + 1);  // IDs 1..N
        REQUIRE(map.insert(orders[i]->id, orders[i]));
    }
    REQUIRE(map.size() == N);

    // Find every order.
    for (std::size_t i = 0; i < N; ++i) {
        Order* found = map.find(static_cast<OrderId>(i + 1));
        REQUIRE(found == orders[i]);
    }

    // Erase all in reverse order.
    for (std::size_t i = N; i > 0; --i) {
        REQUIRE(map.erase(static_cast<OrderId>(i)));
    }
    REQUIRE(map.size() == 0);
    REQUIRE(map.empty());

    // Verify none are findable after erase.
    for (std::size_t i = 0; i < N; ++i) {
        REQUIRE(map.find(static_cast<OrderId>(i + 1)) == nullptr);
    }
}

// ---------------------------------------------------------------------------
// Probe-chain integrity (backward-shift correctness)
// ---------------------------------------------------------------------------

TEST_CASE("OrderMap: backward-shift preserves reachability after erase", "[map]") {
    // This test inserts keys that are likely to collide (sequential IDs
    // stress the hash function), then erases a middle entry and verifies
    // that remaining entries are still reachable.
    OrderPool<TEST_POOL_CAP> pool;
    OrderMap map(TEST_POOL_CAP);

    constexpr std::size_t N = 30;
    Order* orders[N]{};

    for (std::size_t i = 0; i < N; ++i) {
        orders[i] = pool.allocate();
        orders[i]->id = static_cast<OrderId>(i + 1);
        REQUIRE(map.insert(orders[i]->id, orders[i]));
    }

    // Erase every other entry.
    for (std::size_t i = 0; i < N; i += 2) {
        REQUIRE(map.erase(static_cast<OrderId>(i + 1)));
    }

    // Remaining entries must still be findable.
    for (std::size_t i = 1; i < N; i += 2) {
        REQUIRE(map.find(static_cast<OrderId>(i + 1)) == orders[i]);
    }
    // Erased entries must not be findable.
    for (std::size_t i = 0; i < N; i += 2) {
        REQUIRE(map.find(static_cast<OrderId>(i + 1)) == nullptr);
    }
}

TEST_CASE("OrderMap: erase-all then refill", "[map]") {
    OrderPool<TEST_POOL_CAP> pool;
    OrderMap map(TEST_POOL_CAP);

    constexpr std::size_t N = 40;
    Order* orders[N]{};

    // First fill.
    for (std::size_t i = 0; i < N; ++i) {
        orders[i] = pool.allocate();
        orders[i]->id = static_cast<OrderId>(i + 100);
        REQUIRE(map.insert(orders[i]->id, orders[i]));
    }

    // Erase everything.
    for (std::size_t i = 0; i < N; ++i) {
        REQUIRE(map.erase(static_cast<OrderId>(i + 100)));
    }
    REQUIRE(map.empty());

    // Refill with new IDs — must succeed (no tombstone leaks).
    for (std::size_t i = 0; i < N; ++i) {
        REQUIRE(map.insert(static_cast<OrderId>(i + 500), orders[i]));
    }
    REQUIRE(map.size() == N);

    for (std::size_t i = 0; i < N; ++i) {
        REQUIRE(map.find(static_cast<OrderId>(i + 500)) == orders[i]);
    }
}
