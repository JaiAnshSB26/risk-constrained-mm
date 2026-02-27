// ============================================================================
// risk-constrained-mm :: tests/cpp/test_order_queue.cpp
// ============================================================================
// Unit tests for the intrusive doubly-linked OrderQueue.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include "lob/order_queue.hpp"
#include "lob/pool.hpp"

using namespace rcmm;

constexpr std::size_t TEST_POOL_CAP = 64;

// Helper: create an order with given id and qty from the pool.
static Order* make_order(OrderPool<TEST_POOL_CAP>& pool, OrderId id, Qty qty) {
    Order* o = pool.allocate();
    REQUIRE(o != nullptr);
    o->id  = id;
    o->qty = qty;
    return o;
}

TEST_CASE("OrderQueue: empty queue", "[queue]") {
    OrderQueue q;

    REQUIRE(q.empty());
    REQUIRE(q.count()     == 0);
    REQUIRE(q.total_qty() == 0);
    REQUIRE(q.front()     == nullptr);
    REQUIRE(q.back()      == nullptr);
    REQUIRE(q.pop_front() == nullptr);
}

TEST_CASE("OrderQueue: push_back one order", "[queue]") {
    OrderPool<TEST_POOL_CAP> pool;
    OrderQueue q;

    Order* o = make_order(pool, 1, 100);
    q.push_back(o);

    REQUIRE_FALSE(q.empty());
    REQUIRE(q.count()     == 1);
    REQUIRE(q.total_qty() == 100);
    REQUIRE(q.front()     == o);
    REQUIRE(q.back()      == o);
}

TEST_CASE("OrderQueue: FIFO ordering", "[queue]") {
    OrderPool<TEST_POOL_CAP> pool;
    OrderQueue q;

    Order* a = make_order(pool, 1, 10);
    Order* b = make_order(pool, 2, 20);
    Order* c = make_order(pool, 3, 30);

    q.push_back(a);
    q.push_back(b);
    q.push_back(c);

    REQUIRE(q.count()     == 3);
    REQUIRE(q.total_qty() == 60);
    REQUIRE(q.front()     == a);
    REQUIRE(q.back()      == c);

    // Pop in FIFO order.
    REQUIRE(q.pop_front() == a);
    REQUIRE(q.pop_front() == b);
    REQUIRE(q.pop_front() == c);
    REQUIRE(q.empty());
}

TEST_CASE("OrderQueue: remove middle element", "[queue]") {
    OrderPool<TEST_POOL_CAP> pool;
    OrderQueue q;

    Order* a = make_order(pool, 1, 10);
    Order* b = make_order(pool, 2, 20);
    Order* c = make_order(pool, 3, 30);

    q.push_back(a);
    q.push_back(b);
    q.push_back(c);

    q.remove(b);

    REQUIRE(q.count()     == 2);
    REQUIRE(q.total_qty() == 40);
    REQUIRE(q.front()     == a);
    REQUIRE(q.back()      == c);

    // Verify linkage is intact.
    REQUIRE(a->next == c);
    REQUIRE(c->prev == a);

    // Removed node should have null pointers.
    REQUIRE(b->prev == nullptr);
    REQUIRE(b->next == nullptr);
}

TEST_CASE("OrderQueue: remove head", "[queue]") {
    OrderPool<TEST_POOL_CAP> pool;
    OrderQueue q;

    Order* a = make_order(pool, 1, 10);
    Order* b = make_order(pool, 2, 20);

    q.push_back(a);
    q.push_back(b);

    q.remove(a);

    REQUIRE(q.count() == 1);
    REQUIRE(q.front() == b);
    REQUIRE(q.back()  == b);
}

TEST_CASE("OrderQueue: remove tail", "[queue]") {
    OrderPool<TEST_POOL_CAP> pool;
    OrderQueue q;

    Order* a = make_order(pool, 1, 10);
    Order* b = make_order(pool, 2, 20);

    q.push_back(a);
    q.push_back(b);

    q.remove(b);

    REQUIRE(q.count() == 1);
    REQUIRE(q.front() == a);
    REQUIRE(q.back()  == a);
}

TEST_CASE("OrderQueue: remove only element", "[queue]") {
    OrderPool<TEST_POOL_CAP> pool;
    OrderQueue q;

    Order* a = make_order(pool, 1, 10);
    q.push_back(a);
    q.remove(a);

    REQUIRE(q.empty());
    REQUIRE(q.front() == nullptr);
    REQUIRE(q.back()  == nullptr);
}

TEST_CASE("OrderQueue: total_qty tracks partial-fill reduction", "[queue]") {
    OrderPool<TEST_POOL_CAP> pool;
    OrderQueue q;

    Order* o = make_order(pool, 1, 100);
    q.push_back(o);

    REQUIRE(q.total_qty() == 100);

    // Simulate a partial fill of 40.
    o->filled = 40;
    q.reduce_qty(40);

    REQUIRE(q.total_qty() == 60);
}
