// ============================================================================
// risk-constrained-mm :: tests/cpp/test_order_book.cpp
// ============================================================================
// Smoke tests for the OrderBook skeleton (Phase 1).
// Full add/cancel/match tests come in Phases 2 & 3.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include "lob/order_book.hpp"

#include <limits>

using namespace rcmm;

// Small pool for tests.
constexpr std::size_t TEST_POOL_CAP = 128;

TEST_CASE("OrderBook: default construction", "[book]") {
    OrderBook<TEST_POOL_CAP> book;

    // Empty book: best_bid is sentinel-low, best_ask is sentinel-high.
    REQUIRE(book.best_bid() == std::numeric_limits<Price>::min());
    REQUIRE(book.best_ask() == std::numeric_limits<Price>::max());
    // Spread overflows on an empty book; that's by design — callers should
    // check best_bid/best_ask against sentinels before computing spread.
    REQUIRE(book.best_ask() > book.best_bid());
}

TEST_CASE("OrderBook: config is accessible", "[book]") {
    BookConfig cfg;
    cfg.tick_size  = 100;      // 0.01 USDT in tick units
    cfg.base_price = 5'000'000; // 50,000 USDT in tick units

    OrderBook<TEST_POOL_CAP> book(cfg);

    REQUIRE(book.config().tick_size   == 100);
    REQUIRE(book.config().base_price  == 5'000'000);
}

TEST_CASE("OrderBook: price-level lookup returns correct level", "[book]") {
    BookConfig cfg;
    cfg.tick_size  = 1;
    cfg.base_price = 0;

    OrderBook<TEST_POOL_CAP> book(cfg);

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
