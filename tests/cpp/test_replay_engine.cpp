// ============================================================================
// risk-constrained-mm :: tests/cpp/test_replay_engine.cpp
// ============================================================================
// Catch2 test suite for the ReplayEngine (Phase 4).
//
// Covers:
//   * Empty ticks replay
//   * Single add / cancel / modify
//   * Trade ticks generating matches via place_order
//   * Top-of-book tracking
//   * Timestamp advancement
//   * 10-tick integration scenario with full state verification
//   * Parser-to-replay integration test
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include "data/market_data_parser.hpp"
#include "data/replay_engine.hpp"
#include "data/tick.hpp"
#include "lob/order_book.hpp"
#include "lob/types.hpp"

#include <limits>
#include <vector>

using namespace rcmm;

// ── Helpers ─────────────────────────────────────────────────────────────────
namespace {

BookConfig simple_config() {
    return {.tick_size = 1, .base_price = 0, .num_levels = 1024};
}

using Book   = OrderBook<64>;
using Replay = ReplayEngine<64>;

// Convenience builder for a Tick.
Tick make_tick(Timestamp ts, TickAction action, OrderId id,
               Side side, Price price, Qty qty) {
    return Tick{ts, id, price, qty, side, action,
                action == TickAction::Trade};
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════════════
//  Basic operations
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Replay: empty ticks produces no trades and no errors",
          "[replay][basic]") {
    Book book(simple_config());
    Replay engine(book);

    std::vector<Tick> empty;
    auto result = engine.replay(empty);

    CHECK(result.trades.empty());
    CHECK(result.ticks_processed == 0u);
    CHECK(result.errors == 0u);
}

TEST_CASE("Replay: single add updates book",
          "[replay][basic]") {
    Book book(simple_config());
    Replay engine(book);

    std::vector<Tick> ticks = {
        make_tick(1000, TickAction::Add, 1001, Side::Bid, 100, 10),
    };
    auto result = engine.replay(ticks);

    CHECK(result.ticks_processed == 1u);
    CHECK(result.errors == 0u);
    CHECK(book.best_bid() == 100);
    CHECK(book.pool().live_count() == 1u);
}

TEST_CASE("Replay: add then cancel restores empty book",
          "[replay][basic]") {
    Book book(simple_config());
    Replay engine(book);

    std::vector<Tick> ticks = {
        make_tick(1000, TickAction::Add,    1001, Side::Bid, 100, 10),
        make_tick(1001, TickAction::Cancel, 1001, Side::Bid, 100, 10),
    };
    auto result = engine.replay(ticks);

    CHECK(result.ticks_processed == 2u);
    CHECK(result.errors == 0u);
    CHECK(book.pool().live_count() == 0u);
    CHECK(book.best_bid() == std::numeric_limits<Price>::min());
}

TEST_CASE("Replay: cancel non-existent order counts as error",
          "[replay][basic]") {
    Book book(simple_config());
    Replay engine(book);

    std::vector<Tick> ticks = {
        make_tick(1000, TickAction::Cancel, 9999, Side::Bid, 100, 10),
    };
    auto result = engine.replay(ticks);

    CHECK(result.ticks_processed == 0u);
    CHECK(result.errors == 1u);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Modify
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Replay: modify updates order price and qty",
          "[replay][modify]") {
    Book book(simple_config());
    Replay engine(book);

    std::vector<Tick> ticks = {
        make_tick(1000, TickAction::Add,    1001, Side::Ask, 102, 8),
        make_tick(1001, TickAction::Modify, 1001, Side::Ask, 101, 6),
    };
    auto result = engine.replay(ticks);

    CHECK(result.ticks_processed == 2u);
    CHECK(result.errors == 0u);

    // Order should now be at the new price with new qty.
    Order* order = book.find_order(1001);
    REQUIRE(order != nullptr);
    CHECK(order->price == 101);
    CHECK(order->qty == 6);
    CHECK(book.best_ask() == 101);

    // Old level should be empty.
    CHECK(book.level(Side::Ask, 102).queue.empty());
}

// ═══════════════════════════════════════════════════════════════════════════
//  Trade
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Replay: trade tick generates matches via place_order",
          "[replay][trade]") {
    Book book(simple_config());
    Replay engine(book);

    std::vector<Tick> ticks = {
        make_tick(1000, TickAction::Add,   1001, Side::Ask, 100, 5),
        make_tick(1001, TickAction::Add,   1002, Side::Ask, 101, 5),
        make_tick(1002, TickAction::Trade, 2001, Side::Bid, 0,   8),
    };
    auto result = engine.replay(ticks);

    CHECK(result.ticks_processed == 3u);
    REQUIRE(result.trades.size() == 2u);

    // First fill at price 100 (full fill of order 1001).
    CHECK(result.trades[0].maker_id == 1001u);
    CHECK(result.trades[0].taker_id == 2001u);
    CHECK(result.trades[0].price == 100);
    CHECK(result.trades[0].qty == 5);

    // Second fill at price 101 (partial fill of order 1002).
    CHECK(result.trades[1].maker_id == 1002u);
    CHECK(result.trades[1].price == 101);
    CHECK(result.trades[1].qty == 3);

    // Order 1001 fully filled and returned to pool.
    CHECK(book.find_order(1001) == nullptr);

    // Order 1002 partially filled, still in book.
    Order* o1002 = book.find_order(1002);
    REQUIRE(o1002 != nullptr);
    CHECK(o1002->leaves() == 2);
}

TEST_CASE("Replay: trade on empty book produces no trades",
          "[replay][trade]") {
    Book book(simple_config());
    Replay engine(book);

    std::vector<Tick> ticks = {
        make_tick(1000, TickAction::Trade, 2001, Side::Bid, 0, 10),
    };
    auto result = engine.replay(ticks);

    CHECK(result.ticks_processed == 1u);
    CHECK(result.trades.empty());
}

// ═══════════════════════════════════════════════════════════════════════════
//  Top-of-book & timestamp tracking
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Replay: top_of_book tracks state correctly",
          "[replay][tob]") {
    Book book(simple_config());
    Replay engine(book);

    // Initial state: no bids or asks.
    auto tob = engine.top_of_book();
    CHECK(tob.best_bid == std::numeric_limits<Price>::min());
    CHECK(tob.best_ask == std::numeric_limits<Price>::max());
    CHECK(tob.timestamp == 0);

    // Add a bid and an ask.
    std::vector<Tick> ticks = {
        make_tick(5000, TickAction::Add, 1001, Side::Bid, 100, 10),
        make_tick(5001, TickAction::Add, 1002, Side::Ask, 105, 10),
    };
    (void)engine.replay(ticks);

    tob = engine.top_of_book();
    CHECK(tob.best_bid == 100);
    CHECK(tob.best_ask == 105);
    CHECK(tob.timestamp == 5001);
}

TEST_CASE("Replay: current_timestamp advances with ticks",
          "[replay][tob]") {
    Book book(simple_config());
    Replay engine(book);

    CHECK(engine.current_timestamp() == 0);

    std::vector<Tick> ticks = {
        make_tick(1000, TickAction::Add, 1001, Side::Bid, 100, 10),
        make_tick(2000, TickAction::Add, 1002, Side::Ask, 105, 5),
        make_tick(3000, TickAction::Add, 1003, Side::Bid, 99,  5),
    };
    (void)engine.replay(ticks);

    CHECK(engine.current_timestamp() == 3000);
}

// ═══════════════════════════════════════════════════════════════════════════
//  10-tick scenario — full state verification
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Replay: 10-tick scenario with full book state verification",
          "[replay][scenario]") {
    Book book(simple_config());
    Replay engine(book);

    // ──────────────────────────────────────────────────────────────────────
    // Tick timeline:
    //  0: Add  1001 bid@100 qty=10
    //  1: Add  1002 bid@99  qty=5
    //  2: Add  1003 ask@102 qty=8
    //  3: Add  1004 ask@103 qty=12
    //  4: Add  1005 ask@102 qty=3    (same level as 1003)
    //  5: Cancel 1002                (remove bid@99)
    //  6: Modify 1004 ask@101 qty=6  (cancel ask@103, re-add ask@101)
    //  7: Add  1006 bid@100 qty=7    (same level as 1001)
    //  8: Trade 2001 buy qty=10      (sweep ask@101(6) + ask@102(4))
    //  9: Add  1007 ask@104 qty=5
    // ──────────────────────────────────────────────────────────────────────
    std::vector<Tick> ticks = {
        make_tick(1000000, TickAction::Add,    1001, Side::Bid, 100, 10),
        make_tick(1000001, TickAction::Add,    1002, Side::Bid, 99,  5),
        make_tick(1000002, TickAction::Add,    1003, Side::Ask, 102, 8),
        make_tick(1000003, TickAction::Add,    1004, Side::Ask, 103, 12),
        make_tick(1000004, TickAction::Add,    1005, Side::Ask, 102, 3),
        make_tick(1000005, TickAction::Cancel, 1002, Side::Bid, 99,  5),
        make_tick(1000006, TickAction::Modify, 1004, Side::Ask, 101, 6),
        make_tick(1000007, TickAction::Add,    1006, Side::Bid, 100, 7),
        make_tick(1000008, TickAction::Trade,  2001, Side::Bid, 0,   10),
        make_tick(1000009, TickAction::Add,    1007, Side::Ask, 104, 5),
    };

    auto result = engine.replay(ticks);

    // ── Aggregate result ────────────────────────────────────────────────
    CHECK(result.ticks_processed == 10u);
    CHECK(result.errors == 0u);

    // ── Trades from tick 8 (market buy sweep) ───────────────────────────
    REQUIRE(result.trades.size() == 2u);

    // First fill: maker 1004 at ask@101, qty 6 (fully filled).
    CHECK(result.trades[0].maker_id == 1004u);
    CHECK(result.trades[0].taker_id == 2001u);
    CHECK(result.trades[0].price == 101);
    CHECK(result.trades[0].qty == 6);
    CHECK(result.trades[0].taker_side == Side::Bid);

    // Second fill: maker 1003 at ask@102, qty 4 (partially filled).
    CHECK(result.trades[1].maker_id == 1003u);
    CHECK(result.trades[1].taker_id == 2001u);
    CHECK(result.trades[1].price == 102);
    CHECK(result.trades[1].qty == 4);

    // ── Top of book ─────────────────────────────────────────────────────
    CHECK(book.best_bid() == 100);
    CHECK(book.best_ask() == 102);

    // ── Pool integrity ──────────────────────────────────────────────────
    // Live orders: 1001, 1003 (partial), 1005, 1006, 1007 = 5
    CHECK(book.pool().live_count() == 5u);

    // ── Individual order verification ───────────────────────────────────

    // 1001: bid@100, qty=10, untouched.
    Order* o1001 = book.find_order(1001);
    REQUIRE(o1001 != nullptr);
    CHECK(o1001->price == 100);
    CHECK(o1001->qty == 10);
    CHECK(o1001->leaves() == 10);

    // 1002: cancelled — should NOT be in the book.
    CHECK(book.find_order(1002) == nullptr);

    // 1003: ask@102, originally qty=8, partially filled 4 -> leaves 4.
    Order* o1003 = book.find_order(1003);
    REQUIRE(o1003 != nullptr);
    CHECK(o1003->price == 102);
    CHECK(o1003->leaves() == 4);

    // 1004: fully filled in tick 8 — should NOT be in the book.
    CHECK(book.find_order(1004) == nullptr);

    // 1005: ask@102, qty=3, untouched.
    Order* o1005 = book.find_order(1005);
    REQUIRE(o1005 != nullptr);
    CHECK(o1005->price == 102);
    CHECK(o1005->qty == 3);

    // 1006: bid@100, qty=7, untouched.
    Order* o1006 = book.find_order(1006);
    REQUIRE(o1006 != nullptr);
    CHECK(o1006->price == 100);
    CHECK(o1006->qty == 7);

    // 1007: ask@104, qty=5 (added in tick 9).
    Order* o1007 = book.find_order(1007);
    REQUIRE(o1007 != nullptr);
    CHECK(o1007->price == 104);
    CHECK(o1007->qty == 5);

    // ── Level-aggregate quantities ──────────────────────────────────────
    // Bid@100: 1001(10) + 1006(7) = 17
    CHECK(book.level(Side::Bid, 100).queue.total_qty() == 17);

    // Ask@102: 1003(4) + 1005(3) = 7
    CHECK(book.level(Side::Ask, 102).queue.total_qty() == 7);

    // Ask@104: 1007(5)
    CHECK(book.level(Side::Ask, 104).queue.total_qty() == 5);

    // Ask@101: emptied by trade sweep.
    CHECK(book.level(Side::Ask, 101).queue.empty());

    // Ask@103: emptied by modify.
    CHECK(book.level(Side::Ask, 103).queue.empty());

    // ── Timestamp ───────────────────────────────────────────────────────
    CHECK(engine.current_timestamp() == 1000009);
}

// ═══════════════════════════════════════════════════════════════════════════
//  replay_one — step-by-step
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Replay: replay_one processes single tick and returns trades",
          "[replay][step]") {
    Book book(simple_config());
    Replay engine(book);

    // Add a resting ask.
    auto t1 = engine.replay_one(
        make_tick(1000, TickAction::Add, 1001, Side::Ask, 100, 5));
    CHECK(t1.empty());
    CHECK(book.best_ask() == 100);

    // Aggressive buy sweeps it.
    auto t2 = engine.replay_one(
        make_tick(1001, TickAction::Trade, 2001, Side::Bid, 0, 3));
    REQUIRE(t2.size() == 1u);
    CHECK(t2[0].maker_id == 1001u);
    CHECK(t2[0].qty == 3);

    // Remaining order still in book.
    Order* o = book.find_order(1001);
    REQUIRE(o != nullptr);
    CHECK(o->leaves() == 2);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Integration: parser + replay engine
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Replay: integration with MarketDataParser",
          "[replay][integration]") {
    const char* csv =
        "timestamp,action,order_id,side,price,qty\n"
        "1000,add,1001,bid,50,10\n"
        "1001,add,1002,ask,55,8\n"
        "1002,add,1003,ask,53,4\n"
        "1003,trade,2001,bid,0,6\n";

    ParseConfig pcfg{1.0, 1.0};
    auto ticks = MarketDataParser::parse_buffer(csv, pcfg);
    REQUIRE(ticks.size() == 4u);

    Book book(simple_config());
    Replay engine(book);
    auto result = engine.replay(ticks);

    CHECK(result.ticks_processed == 4u);
    CHECK(result.errors == 0u);

    // Trade sweep: buy qty=6 sweeps ask@53(4) then ask@55(2).
    REQUIRE(result.trades.size() == 2u);
    CHECK(result.trades[0].price == 53);
    CHECK(result.trades[0].qty == 4);
    CHECK(result.trades[1].price == 55);
    CHECK(result.trades[1].qty == 2);

    // Book state: bid@50(10), ask@55(6 remaining).
    CHECK(book.best_bid() == 50);
    CHECK(book.best_ask() == 55);
    CHECK(book.level(Side::Ask, 55).queue.total_qty() == 6);
    CHECK(book.level(Side::Ask, 53).queue.empty());
}
