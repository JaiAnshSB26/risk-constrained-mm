// ============================================================================
// risk-constrained-mm :: tests/cpp/test_market_data_parser.cpp
// ============================================================================
// Catch2 test suite for the zero-allocation CSV parser (Phase 4).
//
// All tests use hardcoded string literals — no external files required.
//
// Covers:
//   * Full-buffer parsing with header skip
//   * Individual line parsing (add, cancel, modify, trade)
//   * from_chars precision for prices and quantities
//   * Field extraction (next_field)
//   * Decimal-to-tick / decimal-to-lot conversion helpers
//   * Malformed-line graceful skip
//   * Edge cases: empty buffer, header-only, CRLF line endings
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include "data/market_data_parser.hpp"
#include "data/tick.hpp"
#include "lob/types.hpp"

#include <string_view>

using namespace rcmm;

// ═══════════════════════════════════════════════════════════════════════════
//  next_field helper
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Parser: next_field splits comma-delimited fields",
          "[parser][field]") {
    std::string_view line = "hello,world,42";

    auto f1 = MarketDataParser::next_field(line);
    CHECK(f1 == "hello");
    auto f2 = MarketDataParser::next_field(line);
    CHECK(f2 == "world");
    auto f3 = MarketDataParser::next_field(line);
    CHECK(f3 == "42");
    CHECK(line.empty());
}

TEST_CASE("Parser: next_field handles single field with no comma",
          "[parser][field]") {
    std::string_view line = "only";
    auto f = MarketDataParser::next_field(line);
    CHECK(f == "only");
    CHECK(line.empty());
}

// ═══════════════════════════════════════════════════════════════════════════
//  decimal_to_ticks / decimal_to_lots
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Parser: decimal_to_ticks with tick_size 1.0",
          "[parser][convert]") {
    CHECK(MarketDataParser::decimal_to_ticks(100.0, 1.0) == 100);
    CHECK(MarketDataParser::decimal_to_ticks(0.0, 1.0) == 0);
    CHECK(MarketDataParser::decimal_to_ticks(999.0, 1.0) == 999);
}

TEST_CASE("Parser: decimal_to_ticks with tick_size 0.01",
          "[parser][convert]") {
    CHECK(MarketDataParser::decimal_to_ticks(50000.50, 0.01) == 5000050);
    CHECK(MarketDataParser::decimal_to_ticks(100.00, 0.01) == 10000);
    CHECK(MarketDataParser::decimal_to_ticks(0.01, 0.01) == 1);
}

TEST_CASE("Parser: decimal_to_lots with lot_size 0.001",
          "[parser][convert]") {
    CHECK(MarketDataParser::decimal_to_lots(1.500, 0.001) == 1500);
    CHECK(MarketDataParser::decimal_to_lots(0.001, 0.001) == 1);
    CHECK(MarketDataParser::decimal_to_lots(10.0, 0.001) == 10000);
}

// ═══════════════════════════════════════════════════════════════════════════
//  parse_line — individual action types
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Parser: parse_line parses add bid correctly",
          "[parser][line]") {
    ParseConfig cfg{1.0, 1.0};
    Tick tick{};
    std::string_view line = "1000000,add,1001,bid,100,5";
    REQUIRE(MarketDataParser::parse_line(line, tick, cfg));

    CHECK(tick.timestamp == 1000000);
    CHECK(tick.action == TickAction::Add);
    CHECK(tick.order_id == 1001u);
    CHECK(tick.side == Side::Bid);
    CHECK(tick.price == 100);
    CHECK(tick.qty == 5);
    CHECK_FALSE(tick.is_trade);
}

TEST_CASE("Parser: parse_line parses cancel correctly",
          "[parser][line]") {
    ParseConfig cfg{1.0, 1.0};
    Tick tick{};
    std::string_view line = "2000000,cancel,1002,ask,101,3";
    REQUIRE(MarketDataParser::parse_line(line, tick, cfg));

    CHECK(tick.action == TickAction::Cancel);
    CHECK(tick.order_id == 1002u);
    CHECK(tick.side == Side::Ask);
}

TEST_CASE("Parser: parse_line parses modify correctly",
          "[parser][line]") {
    ParseConfig cfg{1.0, 1.0};
    Tick tick{};
    std::string_view line = "3000000,modify,1003,bid,99,8";
    REQUIRE(MarketDataParser::parse_line(line, tick, cfg));

    CHECK(tick.action == TickAction::Modify);
    CHECK(tick.order_id == 1003u);
    CHECK(tick.price == 99);
    CHECK(tick.qty == 8);
}

TEST_CASE("Parser: parse_line parses trade and sets is_trade",
          "[parser][line]") {
    ParseConfig cfg{1.0, 1.0};
    Tick tick{};
    std::string_view line = "4000000,trade,2001,bid,0,10";
    REQUIRE(MarketDataParser::parse_line(line, tick, cfg));

    CHECK(tick.action == TickAction::Trade);
    CHECK(tick.is_trade);
    CHECK(tick.order_id == 2001u);
    CHECK(tick.qty == 10);
}

// ═══════════════════════════════════════════════════════════════════════════
//  from_chars precision through full parse_line
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Parser: from_chars price precision with tick_size 0.01",
          "[parser][precision]") {
    ParseConfig cfg{0.01, 1.0};
    Tick tick{};
    std::string_view line = "1000000,add,1001,bid,50000.50,5";
    REQUIRE(MarketDataParser::parse_line(line, tick, cfg));
    CHECK(tick.price == 5000050);
}

TEST_CASE("Parser: from_chars qty precision with lot_size 0.001",
          "[parser][precision]") {
    ParseConfig cfg{1.0, 0.001};
    Tick tick{};
    std::string_view line = "1000000,add,1001,bid,100,1.500";
    REQUIRE(MarketDataParser::parse_line(line, tick, cfg));
    CHECK(tick.qty == 1500);
}

TEST_CASE("Parser: from_chars combined price and qty precision",
          "[parser][precision]") {
    ParseConfig cfg{0.01, 0.001};
    Tick tick{};
    std::string_view line = "1000000,add,1001,ask,65432.10,0.025";
    REQUIRE(MarketDataParser::parse_line(line, tick, cfg));
    CHECK(tick.price == 6543210);
    CHECK(tick.qty == 25);
}

// ═══════════════════════════════════════════════════════════════════════════
//  parse_buffer — multi-line
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Parser: parse_buffer with valid multi-line CSV",
          "[parser][buffer]") {
    const char* csv =
        "timestamp,action,order_id,side,price,qty\n"
        "1000000,add,1001,bid,100,10\n"
        "1000001,add,1002,ask,101,5\n"
        "1000002,cancel,1001,bid,100,10\n"
        "1000003,trade,2001,bid,0,5\n";

    ParseConfig cfg{1.0, 1.0};
    auto ticks = MarketDataParser::parse_buffer(csv, cfg);

    REQUIRE(ticks.size() == 4u);

    CHECK(ticks[0].timestamp == 1000000);
    CHECK(ticks[0].action == TickAction::Add);
    CHECK(ticks[0].order_id == 1001u);
    CHECK(ticks[0].side == Side::Bid);
    CHECK(ticks[0].price == 100);
    CHECK(ticks[0].qty == 10);

    CHECK(ticks[1].action == TickAction::Add);
    CHECK(ticks[1].side == Side::Ask);

    CHECK(ticks[2].action == TickAction::Cancel);

    CHECK(ticks[3].action == TickAction::Trade);
    CHECK(ticks[3].is_trade);
}

TEST_CASE("Parser: parse_buffer skips malformed lines gracefully",
          "[parser][buffer]") {
    const char* csv =
        "timestamp,action,order_id,side,price,qty\n"
        "1000000,add,1001,bid,100,10\n"
        "bad_line\n"
        "1000001,INVALID_ACTION,1002,ask,101,5\n"
        "1000002,add,1003,ask,102,3\n";

    ParseConfig cfg{1.0, 1.0};
    auto ticks = MarketDataParser::parse_buffer(csv, cfg);

    // Only 2 valid lines (bad_line and INVALID_ACTION are skipped).
    REQUIRE(ticks.size() == 2u);
    CHECK(ticks[0].order_id == 1001u);
    CHECK(ticks[1].order_id == 1003u);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Edge cases
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Parser: empty buffer returns empty vector",
          "[parser][edge]") {
    ParseConfig cfg{1.0, 1.0};
    auto ticks = MarketDataParser::parse_buffer("", cfg);
    CHECK(ticks.empty());
}

TEST_CASE("Parser: header-only buffer returns empty vector",
          "[parser][edge]") {
    ParseConfig cfg{1.0, 1.0};
    auto ticks = MarketDataParser::parse_buffer(
        "timestamp,action,order_id,side,price,qty\n", cfg);
    CHECK(ticks.empty());
}

TEST_CASE("Parser: handles Windows CRLF line endings",
          "[parser][edge]") {
    const char* csv =
        "timestamp,action,order_id,side,price,qty\r\n"
        "1000000,add,1001,bid,100,10\r\n"
        "1000001,add,1002,ask,101,5\r\n";

    ParseConfig cfg{1.0, 1.0};
    auto ticks = MarketDataParser::parse_buffer(csv, cfg);

    REQUIRE(ticks.size() == 2u);
    CHECK(ticks[0].order_id == 1001u);
    CHECK(ticks[0].price == 100);
    CHECK(ticks[1].order_id == 1002u);
    CHECK(ticks[1].price == 101);
}

TEST_CASE("Parser: malformed line with missing fields returns false",
          "[parser][edge]") {
    ParseConfig cfg{1.0, 1.0};
    Tick tick{};
    // Only 3 fields instead of 6.
    std::string_view line = "1000000,add,1001";
    CHECK_FALSE(MarketDataParser::parse_line(line, tick, cfg));
}

TEST_CASE("Parser: invalid side returns false",
          "[parser][edge]") {
    ParseConfig cfg{1.0, 1.0};
    Tick tick{};
    std::string_view line = "1000000,add,1001,buy,100,5";
    CHECK_FALSE(MarketDataParser::parse_line(line, tick, cfg));
}

TEST_CASE("Parser: non-numeric timestamp returns false",
          "[parser][edge]") {
    ParseConfig cfg{1.0, 1.0};
    Tick tick{};
    std::string_view line = "abc,add,1001,bid,100,5";
    CHECK_FALSE(MarketDataParser::parse_line(line, tick, cfg));
}

TEST_CASE("Parser: buffer without trailing newline parses last line",
          "[parser][edge]") {
    const char* csv =
        "timestamp,action,order_id,side,price,qty\n"
        "1000000,add,1001,bid,100,10";  // no trailing \n

    ParseConfig cfg{1.0, 1.0};
    auto ticks = MarketDataParser::parse_buffer(csv, cfg);
    REQUIRE(ticks.size() == 1u);
    CHECK(ticks[0].order_id == 1001u);
}
