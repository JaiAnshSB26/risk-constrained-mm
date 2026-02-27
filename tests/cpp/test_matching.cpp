// ============================================================================
// risk-constrained-mm :: tests/cpp/test_matching.cpp
// ============================================================================
// Catch2 test suite for the Phase 3 matching engine (place_order).
//
// Covers:
//   • Market orders sweeping single and multiple price levels
//   • Limit orders with partial fills + resting remainder
//   • Price priority (best price matched first)
//   • Time priority (FIFO at each price level)
//   • OrderPool reclamation after full fills
//   • Trade struct field verification
//   • Best bid / ask updates during matching
//   • Edge cases (empty book, exhausted book, exact-limit fills)
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include "lob/order_book.hpp"
#include "lob/types.hpp"

#include <limits>

using namespace rcmm;

// ── Helpers ─────────────────────────────────────────────────────────────────
namespace {

BookConfig simple_config() {
    return {.tick_size = 1, .base_price = 0, .num_levels = 1024};
}

using Book = OrderBook<64>;

}  // namespace

// ═══════════════════════════════════════════════════════════════════════════
//  Market Orders
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Matching: market buy against empty book produces no trades",
          "[matching][market]") {
    Book book(simple_config());
    auto trades = book.place_order(1, 0, 10, Side::Bid, OrderType::Market);
    REQUIRE(trades.empty());
}

TEST_CASE("Matching: market sell against empty book produces no trades",
          "[matching][market]") {
    Book book(simple_config());
    auto trades = book.place_order(1, 0, 10, Side::Ask, OrderType::Market);
    REQUIRE(trades.empty());
}

TEST_CASE("Matching: market buy sweeps single ask level",
          "[matching][market]") {
    Book book(simple_config());
    (void)book.add_order(100, Side::Ask, 50, 10);
    (void)book.add_order(101, Side::Ask, 50, 5);

    auto trades = book.place_order(1, 0, 15, Side::Bid, OrderType::Market);
    REQUIRE(trades.size() == 2);
    CHECK(trades[0].maker_id == 100);
    CHECK(trades[0].qty == 10);
    CHECK(trades[0].price == 50);
    CHECK(trades[1].maker_id == 101);
    CHECK(trades[1].qty == 5);
    CHECK(trades[1].price == 50);

    // Level should be empty, best_ask is sentinel
    CHECK(book.level(Side::Ask, 50).queue.empty());
    CHECK(book.best_ask() == std::numeric_limits<Price>::max());
}

TEST_CASE("Matching: market buy sweeps multiple ask levels",
          "[matching][market]") {
    Book book(simple_config());
    (void)book.add_order(100, Side::Ask, 50, 5);
    (void)book.add_order(101, Side::Ask, 51, 5);
    (void)book.add_order(102, Side::Ask, 52, 5);

    auto trades = book.place_order(1, 0, 12, Side::Bid, OrderType::Market);
    REQUIRE(trades.size() == 3);
    // Price priority: lowest ask (50) first, then 51, then 52
    CHECK(trades[0].price == 50);
    CHECK(trades[0].qty == 5);
    CHECK(trades[1].price == 51);
    CHECK(trades[1].qty == 5);
    CHECK(trades[2].price == 52);
    CHECK(trades[2].qty == 2);   // partial fill

    // best_ask should now be 52 (partially filled level)
    CHECK(book.best_ask() == 52);
    CHECK(book.level(Side::Ask, 52).queue.total_qty() == 3);
}

TEST_CASE("Matching: market sell sweeps multiple bid levels",
          "[matching][market]") {
    Book book(simple_config());
    (void)book.add_order(100, Side::Bid, 52, 5);   // highest bid
    (void)book.add_order(101, Side::Bid, 51, 5);
    (void)book.add_order(102, Side::Bid, 50, 5);

    auto trades = book.place_order(1, 0, 8, Side::Ask, OrderType::Market);
    REQUIRE(trades.size() == 2);
    // Price priority for sells: highest bid (52) first, then 51
    CHECK(trades[0].price == 52);
    CHECK(trades[0].qty == 5);
    CHECK(trades[1].price == 51);
    CHECK(trades[1].qty == 3);

    CHECK(book.best_bid() == 51);
    CHECK(book.level(Side::Bid, 51).queue.total_qty() == 2);
}

TEST_CASE("Matching: market order exhausts entire book",
          "[matching][market]") {
    Book book(simple_config());
    (void)book.add_order(100, Side::Ask, 50, 5);
    (void)book.add_order(101, Side::Ask, 51, 5);

    auto trades = book.place_order(1, 0, 100, Side::Bid, OrderType::Market);
    REQUIRE(trades.size() == 2);
    Qty total_filled = 0;
    for (const auto& t : trades) total_filled += t.qty;
    CHECK(total_filled == 10);

    // Book completely empty
    CHECK(book.best_ask() == std::numeric_limits<Price>::max());
    CHECK(book.pool().live_count() == 0);
}

TEST_CASE("Matching: market sell partially fills single bid",
          "[matching][market]") {
    Book book(simple_config());
    (void)book.add_order(100, Side::Bid, 50, 10);

    auto trades = book.place_order(1, 0, 3, Side::Ask, OrderType::Market);
    REQUIRE(trades.size() == 1);
    CHECK(trades[0].qty == 3);

    // Maker is still on the book with reduced qty
    Order* maker = book.find_order(100);
    REQUIRE(maker != nullptr);
    CHECK(maker->leaves() == 7);
    CHECK(maker->status == OrderStatus::PartialFill);
    CHECK(book.best_bid() == 50);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Limit Orders
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Matching: limit buy fills and rests remainder",
          "[matching][limit]") {
    Book book(simple_config());
    (void)book.add_order(100, Side::Ask, 50, 5);

    auto trades = book.place_order(1, 52, 10, Side::Bid, OrderType::Limit);
    REQUIRE(trades.size() == 1);
    CHECK(trades[0].maker_id == 100);
    CHECK(trades[0].qty == 5);
    CHECK(trades[0].price == 50);

    // Remaining 5 rests as bid at price 52
    Order* resting = book.find_order(1);
    REQUIRE(resting != nullptr);
    CHECK(resting->side == Side::Bid);
    CHECK(resting->price == 52);
    CHECK(resting->qty == 5);
    CHECK(book.best_bid() == 52);
}

TEST_CASE("Matching: limit buy that does not cross rests immediately",
          "[matching][limit]") {
    Book book(simple_config());
    (void)book.add_order(100, Side::Ask, 55, 10);

    auto trades = book.place_order(1, 50, 10, Side::Bid, OrderType::Limit);
    REQUIRE(trades.empty());

    Order* resting = book.find_order(1);
    REQUIRE(resting != nullptr);
    CHECK(resting->price == 50);
    CHECK(resting->qty == 10);
    CHECK(book.best_bid() == 50);
    CHECK(book.best_ask() == 55);
}

TEST_CASE("Matching: limit sell partial fill and rests",
          "[matching][limit]") {
    Book book(simple_config());
    (void)book.add_order(100, Side::Bid, 50, 5);

    auto trades = book.place_order(1, 48, 10, Side::Ask, OrderType::Limit);
    REQUIRE(trades.size() == 1);
    CHECK(trades[0].qty == 5);

    Order* resting = book.find_order(1);
    REQUIRE(resting != nullptr);
    CHECK(resting->side == Side::Ask);
    CHECK(resting->price == 48);
    CHECK(resting->qty == 5);
    CHECK(book.best_ask() == 48);
}

TEST_CASE("Matching: limit buy fills exactly at price limit",
          "[matching][limit]") {
    Book book(simple_config());
    (void)book.add_order(100, Side::Ask, 50, 5);
    (void)book.add_order(101, Side::Ask, 51, 5);

    // Limit price = 50: should only match at 50, not 51
    auto trades = book.place_order(1, 50, 10, Side::Bid, OrderType::Limit);
    REQUIRE(trades.size() == 1);
    CHECK(trades[0].price == 50);
    CHECK(trades[0].qty == 5);

    // Remainder rests at price 50
    Order* resting = book.find_order(1);
    REQUIRE(resting != nullptr);
    CHECK(resting->price == 50);
    CHECK(resting->qty == 5);

    // Ask at 51 untouched
    CHECK(book.best_ask() == 51);
    CHECK(book.level(Side::Ask, 51).queue.total_qty() == 5);
}

TEST_CASE("Matching: limit sell stopped by price limit",
          "[matching][limit]") {
    Book book(simple_config());
    (void)book.add_order(100, Side::Bid, 55, 5);
    (void)book.add_order(101, Side::Bid, 50, 5);

    // Limit sell at price 52: should match at 55 but not at 50
    auto trades = book.place_order(1, 52, 10, Side::Ask, OrderType::Limit);
    REQUIRE(trades.size() == 1);
    CHECK(trades[0].price == 55);
    CHECK(trades[0].qty == 5);

    // Remainder rests as ask at 52
    Order* resting = book.find_order(1);
    REQUIRE(resting != nullptr);
    CHECK(resting->price == 52);
    CHECK(resting->qty == 5);
    CHECK(book.best_ask() == 52);

    // Bid at 50 untouched
    CHECK(book.best_bid() == 50);
    CHECK(book.level(Side::Bid, 50).queue.total_qty() == 5);
}

TEST_CASE("Matching: limit order fully filled is not added to book",
          "[matching][limit]") {
    Book book(simple_config());
    (void)book.add_order(100, Side::Ask, 50, 10);

    auto trades = book.place_order(1, 50, 10, Side::Bid, OrderType::Limit);
    REQUIRE(trades.size() == 1);
    CHECK(trades[0].qty == 10);

    // Order 1 NOT in book (fully filled, never rested)
    CHECK(book.find_order(1) == nullptr);
    CHECK(book.level(Side::Ask, 50).queue.empty());
}

TEST_CASE("Matching: limit on empty book rests immediately",
          "[matching][limit]") {
    Book book(simple_config());

    auto trades = book.place_order(1, 50, 10, Side::Bid, OrderType::Limit);
    REQUIRE(trades.empty());

    Order* resting = book.find_order(1);
    REQUIRE(resting != nullptr);
    CHECK(resting->price == 50);
    CHECK(resting->qty == 10);
    CHECK(book.best_bid() == 50);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Time Priority (FIFO)
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Matching: time priority - older order filled first",
          "[matching][fifo]") {
    Book book(simple_config());
    (void)book.add_order(100, Side::Ask, 50, 3);  // oldest
    (void)book.add_order(101, Side::Ask, 50, 3);  // middle
    (void)book.add_order(102, Side::Ask, 50, 3);  // newest

    auto trades = book.place_order(1, 50, 5, Side::Bid, OrderType::Limit);
    REQUIRE(trades.size() == 2);
    // Oldest (100) fully filled first
    CHECK(trades[0].maker_id == 100);
    CHECK(trades[0].qty == 3);
    // Middle (101) partially filled next
    CHECK(trades[1].maker_id == 101);
    CHECK(trades[1].qty == 2);

    // 100 removed from book
    CHECK(book.find_order(100) == nullptr);

    // 101 still resting with 1 remaining
    Order* o101 = book.find_order(101);
    REQUIRE(o101 != nullptr);
    CHECK(o101->leaves() == 1);
    CHECK(o101->status == OrderStatus::PartialFill);

    // 102 untouched
    Order* o102 = book.find_order(102);
    REQUIRE(o102 != nullptr);
    CHECK(o102->leaves() == 3);
    CHECK(o102->status == OrderStatus::New);
}

TEST_CASE("Matching: time priority on bid side",
          "[matching][fifo]") {
    Book book(simple_config());
    (void)book.add_order(200, Side::Bid, 50, 4);  // oldest
    (void)book.add_order(201, Side::Bid, 50, 4);  // newest

    auto trades = book.place_order(1, 50, 6, Side::Ask, OrderType::Limit);
    REQUIRE(trades.size() == 2);
    CHECK(trades[0].maker_id == 200);
    CHECK(trades[0].qty == 4);   // fully filled (FIFO)
    CHECK(trades[1].maker_id == 201);
    CHECK(trades[1].qty == 2);   // partially filled

    Order* o201 = book.find_order(201);
    REQUIRE(o201 != nullptr);
    CHECK(o201->leaves() == 2);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Pool Reclamation
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Matching: pool reclaims memory after full fill",
          "[matching][pool]") {
    Book book(simple_config());
    auto before = book.pool().live_count();
    (void)book.add_order(100, Side::Ask, 50, 10);
    CHECK(book.pool().live_count() == before + 1);

    auto trades = book.place_order(1, 0, 10, Side::Bid, OrderType::Market);
    REQUIRE(trades.size() == 1);
    CHECK(book.pool().live_count() == before);   // returned to pool
    CHECK(book.find_order(100) == nullptr);       // erased from map
}

TEST_CASE("Matching: pool reclaims across multi-level sweep",
          "[matching][pool]") {
    Book book(simple_config());
    (void)book.add_order(100, Side::Ask, 50, 5);
    (void)book.add_order(101, Side::Ask, 51, 5);
    (void)book.add_order(102, Side::Ask, 52, 5);
    CHECK(book.pool().live_count() == 3);

    auto trades = book.place_order(1, 0, 15, Side::Bid, OrderType::Market);
    REQUIRE(trades.size() == 3);
    CHECK(book.pool().live_count() == 0);   // all 3 returned to pool
}

// ═══════════════════════════════════════════════════════════════════════════
//  Trade Struct Fields
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Matching: trade fields are populated correctly",
          "[matching][trade]") {
    Book book(simple_config());
    (void)book.add_order(100, Side::Ask, 50, 10);

    auto trades = book.place_order(42, 0, 5, Side::Bid, OrderType::Market, 999);
    REQUIRE(trades.size() == 1);
    CHECK(trades[0].maker_id == 100);
    CHECK(trades[0].taker_id == 42);
    CHECK(trades[0].price == 50);
    CHECK(trades[0].qty == 5);
    CHECK(trades[0].taker_side == Side::Bid);
    CHECK(trades[0].timestamp == 999);
}

TEST_CASE("Matching: sell-side trade has correct taker_side",
          "[matching][trade]") {
    Book book(simple_config());
    (void)book.add_order(100, Side::Bid, 50, 10);

    auto trades = book.place_order(42, 0, 5, Side::Ask, OrderType::Market, 777);
    REQUIRE(trades.size() == 1);
    CHECK(trades[0].taker_side == Side::Ask);
    CHECK(trades[0].timestamp == 777);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Best Bid / Ask Updates
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Matching: best_ask updates after sweeping a level",
          "[matching][best]") {
    Book book(simple_config());
    (void)book.add_order(100, Side::Ask, 50, 5);
    (void)book.add_order(101, Side::Ask, 55, 5);
    CHECK(book.best_ask() == 50);

    auto trades = book.place_order(1, 0, 5, Side::Bid, OrderType::Market);
    REQUIRE(trades.size() == 1);
    CHECK(book.best_ask() == 55);
}

TEST_CASE("Matching: best_bid updates after sweeping a level",
          "[matching][best]") {
    Book book(simple_config());
    (void)book.add_order(100, Side::Bid, 55, 5);
    (void)book.add_order(101, Side::Bid, 50, 5);
    CHECK(book.best_bid() == 55);

    auto trades = book.place_order(1, 0, 5, Side::Ask, OrderType::Market);
    REQUIRE(trades.size() == 1);
    CHECK(book.best_bid() == 50);
}

TEST_CASE("Matching: best_bid set by limit remainder after matching",
          "[matching][best]") {
    Book book(simple_config());
    (void)book.add_order(100, Side::Ask, 50, 5);

    // Limit buy at 52, fills 5 at ask 50, rests 5 at bid 52
    auto trades = book.place_order(1, 52, 10, Side::Bid, OrderType::Limit);
    REQUIRE(trades.size() == 1);
    CHECK(book.best_bid() == 52);
    CHECK(book.best_ask() == std::numeric_limits<Price>::max());
}

// ═══════════════════════════════════════════════════════════════════════════
//  Stress / Integration
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Matching: limit buy sweeps multiple levels then rests",
          "[matching][stress]") {
    Book book(simple_config());
    // Seed asks at prices 51..60 with qty = index
    for (OrderId i = 1; i <= 10; ++i) {
        (void)book.add_order(i, Side::Ask,
                             50 + static_cast<Price>(i),
                             static_cast<Qty>(i));
    }

    // Aggressive limit buy at price 53, qty 15
    // Should fill: 51(1) + 52(2) + 53(3) = 6, rest 9 at price 53
    auto trades = book.place_order(100, 53, 15, Side::Bid, OrderType::Limit);
    Qty total_filled = 0;
    for (const auto& t : trades) total_filled += t.qty;
    CHECK(total_filled == 6);
    REQUIRE(trades.size() == 3);
    CHECK(trades[0].price == 51);
    CHECK(trades[1].price == 52);
    CHECK(trades[2].price == 53);

    // Remainder rests at 53
    Order* resting = book.find_order(100);
    REQUIRE(resting != nullptr);
    CHECK(resting->qty == 9);
    CHECK(resting->price == 53);

    // best_ask moved to 54, best_bid set to 53
    CHECK(book.best_ask() == 54);
    CHECK(book.best_bid() == 53);
}

TEST_CASE("Matching: alternating buy/sell limit orders build and consume book",
          "[matching][stress]") {
    Book book(simple_config());

    // Phase A: seed 5 asks at distinct levels
    for (OrderId i = 1; i <= 5; ++i) {
        (void)book.add_order(i, Side::Ask,
                             100 + static_cast<Price>(i), 10);
    }
    CHECK(book.pool().live_count() == 5);
    CHECK(book.best_ask() == 101);

    // Phase B: aggressive limit buy sweeps first 3 levels, rests at 103
    auto t1 = book.place_order(50, 103, 35, Side::Bid, OrderType::Limit);
    Qty filled_b = 0;
    for (const auto& t : t1) filled_b += t.qty;
    CHECK(filled_b == 30);  // 10+10+10 at 101,102,103

    Order* resting_bid = book.find_order(50);
    REQUIRE(resting_bid != nullptr);
    CHECK(resting_bid->qty == 5);
    CHECK(resting_bid->price == 103);
    CHECK(book.best_bid() == 103);
    CHECK(book.best_ask() == 104);

    // Phase C: aggressive limit sell that crosses the resting bid
    auto t2 = book.place_order(60, 100, 8, Side::Ask, OrderType::Limit);
    REQUIRE(t2.size() == 1);
    CHECK(t2[0].maker_id == 50);
    CHECK(t2[0].qty == 5);
    CHECK(t2[0].price == 103);

    // Remaining 3 rests as ask at 100
    Order* resting_ask = book.find_order(60);
    REQUIRE(resting_ask != nullptr);
    CHECK(resting_ask->price == 100);
    CHECK(resting_ask->qty == 3);
    CHECK(book.best_ask() == 100);
}

TEST_CASE("Matching: large sweep with pool integrity check",
          "[matching][stress]") {
    Book book(simple_config());

    // Add 30 orders across 10 price levels (3 per level)
    OrderId oid = 1;
    for (Price p = 100; p < 110; ++p) {
        for (int j = 0; j < 3; ++j) {
            (void)book.add_order(oid++, Side::Ask, p, 5);
        }
    }
    CHECK(book.pool().live_count() == 30);
    CHECK(book.best_ask() == 100);

    // Market buy sweeps everything
    auto trades = book.place_order(500, 0, 1000, Side::Bid, OrderType::Market);
    Qty total = 0;
    for (const auto& t : trades) total += t.qty;
    CHECK(total == 150);   // 30 orders × 5 qty each
    CHECK(book.pool().live_count() == 0);
    CHECK(book.best_ask() == std::numeric_limits<Price>::max());
}
