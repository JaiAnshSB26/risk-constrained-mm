// ============================================================================
// risk-constrained-mm :: cpp/include/data/market_data_parser.hpp
// ============================================================================
// High-performance, zero-allocation (parsing path) CSV parser for tick data.
//
// Design constraints:
//   * NO std::stringstream, NO std::getline allocating std::string.
//   * File is read into a single contiguous buffer (fread / mmap).
//   * Lines are located via simple '\n' scan on std::string_view.
//   * Fields are extracted via ',' scan — zero-copy (string_view::substr).
//   * Numeric conversions use std::from_chars (no locale, no allocation).
//   * Output: std::vector<Tick> with a single reserve() allocation.
//
// Expected CSV format (first line = header, skipped):
//   timestamp,action,order_id,side,price,qty
//   1609459200000000,add,12345,bid,50000.50,1.500
//
// ============================================================================
#pragma once

#include "data/tick.hpp"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <string_view>
#include <system_error>
#include <vector>

namespace rcmm {

/// Configuration for decimal-to-integer conversions during parsing.
struct ParseConfig {
    double tick_size = 1.0;   ///< decimal price units per tick (e.g. 0.01)
    double lot_size  = 1.0;   ///< decimal qty units per lot   (e.g. 0.001)
};

/// High-performance, zero-allocation (parsing path) CSV parser.
///
/// Accepts a contiguous buffer (from fread, mmap, or a string literal) and
/// returns a vector of Tick structs.  All per-row work is zero-allocation:
/// std::string_view slicing + std::from_chars integer/float conversion.
class MarketDataParser {
public:
    // ── Main API ────────────────────────────────────────────────────────────

    /// Parse an entire buffer of CSV tick data.
    /// The first line is treated as a header and skipped.
    [[nodiscard]] static std::vector<Tick> parse_buffer(
            std::string_view data, const ParseConfig& cfg) {
        std::vector<Tick> ticks;
        ticks.reserve(estimate_line_count(data));

        // Skip header line.
        auto nl = data.find('\n');
        if (nl == std::string_view::npos) return ticks;
        data.remove_prefix(nl + 1u);

        // Parse each subsequent line.
        while (!data.empty()) {
            auto line = next_line(data);
            if (line.empty()) continue;

            Tick tick{};
            if (parse_line(line, tick, cfg)) {
                ticks.push_back(tick);
            }
        }

        return ticks;
    }

    /// Parse a single CSV line into a Tick.
    /// Returns true on success, false if the line is malformed.
    [[nodiscard]] static bool parse_line(
            std::string_view line, Tick& out,
            const ParseConfig& cfg) noexcept {
        // Strip trailing \r (Windows line endings).
        if (!line.empty() && line.back() == '\r') {
            line.remove_suffix(1u);
        }
        if (line.empty()) return false;

        // Field 0: timestamp (int64 nanoseconds).
        auto ts_field = next_field(line);
        if (!parse_int64(ts_field, out.timestamp)) return false;

        // Field 1: action (add | cancel | modify | trade).
        auto action_field = next_field(line);
        if (!parse_action(action_field, out.action)) return false;
        out.is_trade = (out.action == TickAction::Trade);

        // Field 2: order_id (uint64).
        auto id_field = next_field(line);
        if (!parse_uint64(id_field, out.order_id)) return false;

        // Field 3: side (bid | ask).
        auto side_field = next_field(line);
        if (!parse_side(side_field, out.side)) return false;

        // Field 4: price (decimal -> tick units via from_chars + llround).
        auto price_field = next_field(line);
        double price_raw = 0.0;
        if (!parse_double(price_field, price_raw)) return false;
        out.price = decimal_to_ticks(price_raw, cfg.tick_size);

        // Field 5: qty (decimal -> lot units).
        auto qty_field = next_field(line);
        if (qty_field.empty()) return false;
        double qty_raw = 0.0;
        if (!parse_double(qty_field, qty_raw)) return false;
        out.qty = decimal_to_lots(qty_raw, cfg.lot_size);

        return true;
    }

    /// Load an entire file into a byte buffer.  Returns empty on failure.
    [[nodiscard]] static std::vector<char> load_file(const char* path) {
        std::vector<char> buf;
        std::FILE* f = std::fopen(path, "rb");
        if (f == nullptr) return buf;

        if (std::fseek(f, 0, SEEK_END) != 0) {
            std::fclose(f);
            return buf;
        }
        long size = std::ftell(f);
        if (size <= 0) { std::fclose(f); return buf; }

        std::fseek(f, 0, SEEK_SET);
        buf.resize(static_cast<std::size_t>(size));
        auto read = std::fread(buf.data(), 1u, buf.size(), f);
        std::fclose(f);
        buf.resize(read);
        return buf;
    }

    // ── Field helpers (public for unit testing) ─────────────────────────────

    /// Extract the next comma-delimited field and advance `line` past it.
    [[nodiscard]] static std::string_view next_field(
            std::string_view& line) noexcept {
        auto pos = line.find(',');
        std::string_view field;
        if (pos == std::string_view::npos) {
            field = line;
            line  = {};
        } else {
            field = line.substr(0u, pos);
            line.remove_prefix(pos + 1u);
        }
        return field;
    }

    /// Convert a decimal price to integer tick units.
    [[nodiscard]] static Price decimal_to_ticks(
            double value, double tick_sz) noexcept {
        return static_cast<Price>(std::llround(value / tick_sz));
    }

    /// Convert a decimal quantity to integer lot units.
    [[nodiscard]] static Qty decimal_to_lots(
            double value, double lot_sz) noexcept {
        return static_cast<Qty>(std::llround(value / lot_sz));
    }

private:
    // ── Line extraction ─────────────────────────────────────────────────────

    /// Extract the next line from `data` (up to \n) and advance past it.
    [[nodiscard]] static std::string_view next_line(
            std::string_view& data) noexcept {
        auto nl = data.find('\n');
        std::string_view line;
        if (nl == std::string_view::npos) {
            line = data;
            data = {};
        } else {
            line = data.substr(0u, nl);
            data.remove_prefix(nl + 1u);
        }
        // Strip trailing \r.
        if (!line.empty() && line.back() == '\r') {
            line.remove_suffix(1u);
        }
        return line;
    }

    /// Rough estimate of line count for reserve().
    [[nodiscard]] static std::size_t estimate_line_count(
            std::string_view data) noexcept {
        // Assume ~40 bytes per CSV line on average.
        return data.size() / 40u + 1u;
    }

    // ── Numeric parsing (std::from_chars — zero allocation) ─────────────────

    [[nodiscard]] static bool parse_int64(
            std::string_view field, Timestamp& out) noexcept {
        if (field.empty()) return false;
        auto [ptr, ec] = std::from_chars(
            field.data(), field.data() + field.size(), out);
        return ec == std::errc{};
    }

    [[nodiscard]] static bool parse_uint64(
            std::string_view field, OrderId& out) noexcept {
        if (field.empty()) return false;
        auto [ptr, ec] = std::from_chars(
            field.data(), field.data() + field.size(), out);
        return ec == std::errc{};
    }

    [[nodiscard]] static bool parse_double(
            std::string_view field, double& out) noexcept {
        if (field.empty()) return false;
        auto [ptr, ec] = std::from_chars(
            field.data(), field.data() + field.size(), out);
        return ec == std::errc{};
    }

    // ── Enum parsing ────────────────────────────────────────────────────────

    [[nodiscard]] static bool parse_side(
            std::string_view field, Side& out) noexcept {
        if (field == "bid") { out = Side::Bid; return true; }
        if (field == "ask") { out = Side::Ask; return true; }
        return false;
    }

    [[nodiscard]] static bool parse_action(
            std::string_view field, TickAction& out) noexcept {
        if (field == "add")    { out = TickAction::Add;    return true; }
        if (field == "cancel") { out = TickAction::Cancel; return true; }
        if (field == "modify") { out = TickAction::Modify; return true; }
        if (field == "trade")  { out = TickAction::Trade;  return true; }
        return false;
    }
};

}  // namespace rcmm
