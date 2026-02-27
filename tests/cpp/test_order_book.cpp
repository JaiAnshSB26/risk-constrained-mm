// ============================================================================
// risk-constrained-mm :: tests/cpp/test_order_book.cpp
// ============================================================================
// Unit tests for the OrderBook — skeleton (Phase 1) and API (Phase 2).
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include "lob/order_book.hpp"

#include <limits>

using namespace rcmm;

// Small pool for tests — keeps allocations stack-friendly.
constexpr std::size_t TEST_POOL_CAP = 128;

// Helper: build a book with tick_size=1, base_price=0, 1024 levels.
static BookConfig simple_config() {
    BookConfig cfg;
    cfg.tick_size  = 1;
    cfg.base_price = 0;
    cfg.num_levels = 1024;
    return cfg;
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase 1 — Skeleton smoke tests
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("OrderBook: default construction", "[book]") {
    OrderBook<TEST_POOL_CAP> book;

    REQUIRE(book.best_bid() == std::numeric_limits<Price>::min());
    REQUIRE(book.best_ask() == std::numeric_limits<Price>::max());
    REQUIRE(book.best_ask() > book.best_bid());
}

TEST_CASE("OrderBook: config is accessible", "[book]") {
    BookConfig cfg;
    cfg.tick_size  = 100;
    cfg.base_price = 5'000'000;

    OrderBook<TEST_POOL_CAP> book(cfg);

    REQUIRE(book.config().tick_size   == 100);
    REQUIRE(book.config().base_price  == 5'000'000);
}

TEST_CASE("OrderBook: price-level lookup returns correct level", "[book]") {
    OrderBook<TEST_POOL_CAP> book(simple_config());

    auto& bid_50 = book.level(Side::Bid, 50);
    REQUIRE(bid_50.price == 50);
    REQUIRE(bid_50.empty());

    auto& ask_100 = book.level(Side::Ask, 100);
    REQUIRE(ask_100.price == 100);
    REQUIRE(ask_100.empty());
}

TEST_CASE("OrderBook: pool is embedded and functional", "[book]") {
    OrderBook<TEST_POOL_CAP> book;

    REQUIRE(book.pool().capacity()   == TEST_POOL_CAP);
    REQUIRE(book.pool().live_count() == 0);
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase 2 — add_order
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("OrderBook: add_order places order on correct level", "[book][add]") {
    OrderBook<TEST_POOL_CAP> book(simple_config());

    Order* o = book.add_order(1, Side::Bid, 100, 50);
    REQUIRE(o != nullptr);
    REQUIRE(o->id    == 1);
    REQUIRE(o->side  == Side::Bid);
    REQUIRE(o->price == 100);
    REQUIRE(o->qty   == 50);

    auto& lvl = book.level(Side::Bid, 100);
    REQUIRE(lvl.queue.count()     == 1);
    REQUIRE(lvl.queue.total_qty() == 50);
}

TEST_CASE("OrderBook: add_order updates pool live count", "[book][add]") {
    OrderBook<TEST_POOL_CAP> book(simple_config());

    REQUIRE(book.pool().live_count() == 0);
    (void)book.add_order(1, Side::Bid, 100, 10);
    REQUIRE(book.pool().live_count() == 1);
    (void)book.add_order(2, Side::Ask, 200, 20);
    REQUIRE(book.pool().live_count() == 2);
}

TEST_CASE("OrderBook: add_order registers in order map", "[book][add]") {
    OrderBook<TEST_POOL_CAP> book(simple_config());

    Order* o = book.add_order(42, Side::Bid, 100, 10);
    REQUIRE(book.order_map().size() == 1);
    REQUIRE(book.find_order(42) == o);
}

TEST_CASE("OrderBook: add_order rejects duplicate ID", "[book][add]") {
    OrderBook<TEST_POOL_CAP> book(simple_config());

    REQUIRE(book.add_order(1, Side::Bid, 100, 10) != nullptr);
    REQUIRE(book.add_order(1, Side::Ask, 200, 20) == nullptr);  // dup

    // Pool slot was rolled back — only 1 live order.
    REQUIRE(book.pool().live_count() == 1);
    REQUIRE(book.order_map().size()  == 1);
}

TEST_CASE("OrderBook: add_order updates best_bid", "[book][add]") {
    OrderBook<TEST_POOL_CAP> book(simple_config());

    (void)book.add_order(1, Side::Bid, 100, 10);
    REQUIRE(book.best_bid() == 100);

    (void)book.add_order(2, Side::Bid, 150, 10);
    REQUIRE(book.best_bid() == 150);

    // Adding a lower bid should NOT change best_bid.
    (void)book.add_order(3, Side::Bid, 90, 10);
    REQUIRE(book.best_bid() == 150);
}

TEST_CASE("OrderBook: add_order updates best_ask", "[book][add]") {
    OrderBook<TEST_POOL_CAP> book(simple_config());

    (void)book.add_order(1, Side::Ask, 500, 10);
    REQUIRE(book.best_ask() == 500);

    (void)book.add_order(2, Side::Ask, 400, 10);
    REQUIRE(book.best_ask() == 400);

    // Adding a higher ask should NOT change best_ask.
    (void)book.add_order(3, Side::Ask, 600, 10);
    REQUIRE(book.best_ask() == 400);
}

TEST_CASE("OrderBook: multiple orders at same level aggregate qty", "[book][add]") {
    OrderBook<TEST_POOL_CAP> book(simple_config());

    (void)book.add_order(1, Side::Bid, 100, 30);
    (void)book.add_order(2, Side::Bid, 100, 70);

    auto& lvl = book.level(Side::Bid, 100);
    REQUIRE(lvl.queue.count()     == 2);
    REQUIRE(lvl.queue.total_qty() == 100);
}

TEST_CASE("OrderBook: add_order assigns monotonic sequence numbers", "[book][add]") {
    OrderBook<TEST_POOL_CAP> book(simple_config());

    Order* a = book.add_order(1, Side::Bid, 100, 10);
    Order* b = book.add_order(2, Side::Bid, 100, 20);
    Order* c = book.add_order(3, Side::Ask, 200, 30);

    REQUIRE(a->seq < b->seq);
    REQUIRE(b->seq < c->seq);
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase 2 — cancel_order
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("OrderBook: cancel_order removes from level and returns to pool", "[book][cancel]") {
    OrderBook<TEST_POOL_CAP> book(simple_config());

    (void)book.add_order(1, Side::Bid, 100, 50);
    REQUIRE(book.pool().live_count() == 1);

    REQUIRE(book.cancel_order(1));

    REQUIRE(book.pool().live_count()  == 0);
    REQUIRE(book.order_map().size()   == 0);
    REQUIRE(book.find_order(1)        == nullptr);

    auto& lvl = book.level(Side::Bid, 100);
    REQUIRE(lvl.queue.count()     == 0);
    REQUIRE(lvl.queue.total_qty() == 0);
}

TEST_CASE("OrderBook: cancel non-existent order returns false", "[book][cancel]") {
    OrderBook<TEST_POOL_CAP> book(simple_config());

    REQUIRE_FALSE(book.cancel_order(999));
    // State unchanged.
    REQUIRE(book.pool().live_count() == 0);
}

TEST_CASE("OrderBook: cancel middle order preserves others", "[book][cancel]") {
    OrderBook<TEST_POOL_CAP> book(simple_config());

    (void)book.add_order(1, Side::Bid, 100, 10);
    (void)book.add_order(2, Side::Bid, 100, 20);
    (void)book.add_order(3, Side::Bid, 100, 30);

    REQUIRE(book.cancel_order(2));

    auto& lvl = book.level(Side::Bid, 100);
    REQUIRE(lvl.queue.count()     == 2);
    REQUIRE(lvl.queue.total_qty() == 40);  // 10 + 30

    // FIFO: front should be order 1, back should be order 3.
    REQUIRE(lvl.queue.front()->id == 1);
    REQUIRE(lvl.queue.back()->id  == 3);
}

TEST_CASE("OrderBook: cancel updates best_bid when level empties", "[book][cancel]") {
    OrderBook<TEST_POOL_CAP> book(simple_config());

    (void)book.add_order(1, Side::Bid, 100, 10);
    (void)book.add_order(2, Side::Bid, 150, 10);
    REQUIRE(book.best_bid() == 150);

    // Cancel the order at the best price — best should fall to 100.
    REQUIRE(book.cancel_order(2));
    REQUIRE(book.best_bid() == 100);
}

TEST_CASE("OrderBook: cancel updates best_ask when level empties", "[book][cancel]") {
    OrderBook<TEST_POOL_CAP> book(simple_config());

    (void)book.add_order(1, Side::Ask, 300, 10);
    (void)book.add_order(2, Side::Ask, 200, 10);
    REQUIRE(book.best_ask() == 200);

    // Cancel the order at the best price — best should rise to 300.
    REQUIRE(book.cancel_order(2));
    REQUIRE(book.best_ask() == 300);
}

TEST_CASE("OrderBook: cancel all bids resets best_bid to sentinel", "[book][cancel]") {
    OrderBook<TEST_POOL_CAP> book(simple_config());

    (void)book.add_order(1, Side::Bid, 100, 10);
    (void)book.add_order(2, Side::Bid, 100, 20);
    REQUIRE(book.best_bid() == 100);

    book.cancel_order(1);
    book.cancel_order(2);
    REQUIRE(book.best_bid() == std::numeric_limits<Price>::min());
}

TEST_CASE("OrderBook: cancel all asks resets best_ask to sentinel", "[book][cancel]") {
    OrderBook<TEST_POOL_CAP> book(simple_config());

    (void)book.add_order(1, Side::Ask, 300, 10);
    REQUIRE(book.best_ask() == 300);

    book.cancel_order(1);
    REQUIRE(book.best_ask() == std::numeric_limits<Price>::max());
}

TEST_CASE("OrderBook: cancel then re-add same ID succeeds", "[book][cancel]") {
    OrderBook<TEST_POOL_CAP> book(simple_config());

    (void)book.add_order(1, Side::Bid, 100, 10);
    book.cancel_order(1);

    // Re-add with the same ID — must succeed.
    Order* o = book.add_order(1, Side::Ask, 200, 30);
    REQUIRE(o != nullptr);
    REQUIRE(o->side  == Side::Ask);
    REQUIRE(o->price == 200);
    REQUIRE(o->qty   == 30);
}

TEST_CASE("OrderBook: cancel does not affect best when level still has orders", "[book][cancel]") {
    OrderBook<TEST_POOL_CAP> book(simple_config());

    (void)book.add_order(1, Side::Bid, 150, 10);
    (void)book.add_order(2, Side::Bid, 150, 20);
    REQUIRE(book.best_bid() == 150);

    // Cancel one — level is not empty, best_bid stays.
    book.cancel_order(1);
    REQUIRE(book.best_bid() == 150);
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase 2 — Full round-trip (add + cancel) stress
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("OrderBook: add and cancel many orders, verify pool integrity", "[book][stress]") {
    OrderBook<TEST_POOL_CAP> book(simple_config());

    constexpr std::size_t N = 60;

    // Add N orders across different price levels.
    for (std::size_t i = 0; i < N; ++i) {
        auto id    = static_cast<OrderId>(i + 1);
        auto price = static_cast<Price>(50 + (i % 20));  // 20 distinct levels
        auto side  = (i % 2 == 0) ? Side::Bid : Side::Ask;
        REQUIRE(book.add_order(id, side, price, 10) != nullptr);
    }
    REQUIRE(book.pool().live_count() == N);
    REQUIRE(book.order_map().size()  == N);

    // Cancel all.
    for (std::size_t i = 0; i < N; ++i) {
        REQUIRE(book.cancel_order(static_cast<OrderId>(i + 1)));
    }
    REQUIRE(book.pool().live_count() == 0);
    REQUIRE(book.order_map().size()  == 0);
    REQUIRE(book.order_map().empty());

    // Re-add — pool and map slots should be fully reusable.
    for (std::size_t i = 0; i < N; ++i) {
        auto id    = static_cast<OrderId>(i + 1000);
        auto price = static_cast<Price>(100 + (i % 10));
        REQUIRE(book.add_order(id, Side::Bid, price, 5) != nullptr);
    }
    REQUIRE(book.pool().live_count() == N);
    REQUIRE(book.order_map().size()  == N);
}
