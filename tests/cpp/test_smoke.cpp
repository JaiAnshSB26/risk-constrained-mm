// ============================================================================
// risk-constrained-mm :: tests/cpp/test_smoke.cpp
// ============================================================================
// Phase 1 smoke test: verifies that Catch2 links, compiles, and runs.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Catch2 framework is operational", "[smoke]") {
    REQUIRE(1 + 1 == 2);
}
