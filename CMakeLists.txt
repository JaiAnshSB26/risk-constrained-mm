# ============================================================================
# risk-constrained-mm :: Root CMakeLists.txt
# ============================================================================
# Minimum 3.20 for C++20 support & modern FetchContent behaviour
cmake_minimum_required(VERSION 3.20)

project(
    risk_constrained_mm
    VERSION   0.1.0
    LANGUAGES CXX
    DESCRIPTION "High-frequency LOB matching engine with risk-constrained RL"
)

# ── Global compiler settings ─────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)        # no GNU extensions — strict ISO C++20
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # for clangd / IDE integration

# Aggressive warning flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall -Wextra -Wpedantic -Werror
        -Wconversion -Wsign-conversion -Wshadow
        -Wnon-virtual-dtor -Wold-style-cast -Wcast-align
        -Woverloaded-virtual -Wformat=2
        -fno-exceptions    # hot-path must never throw
    )
    # Release-mode: enable LTO and architecture-native tuning
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -march=native -flto)
        add_link_options(-flto)
    endif()
elseif(MSVC)
    add_compile_options(/W4 /WX /permissive- /Zc:__cplusplus)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(/O2 /GL)
        add_link_options(/LTCG)
    endif()
endif()

# ── FetchContent: external dependencies ──────────────────────────────────────
include(FetchContent)

# Catch2 v3 — header-only test framework
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.2
    GIT_SHALLOW    TRUE
)

# pybind11 — zero-overhead Python ↔ C++ bindings
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.12.0
    GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(Catch2 pybind11)

# ── Core static library ─────────────────────────────────────────────────────
# All LOB / engine / simulation sources get compiled into one static lib.
# Headers are in cpp/include/; sources will go in cpp/src/.
file(GLOB_RECURSE LIB_SOURCES CONFIGURE_DEPENDS "cpp/src/*.cpp")

add_library(lob_core STATIC ${LIB_SOURCES})
target_include_directories(lob_core PUBLIC cpp/include)
target_compile_features(lob_core PUBLIC cxx_std_20)

# ── Testing ──────────────────────────────────────────────────────────────────
enable_testing()

file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS "tests/cpp/*.cpp")

add_executable(unit_tests ${TEST_SOURCES})
target_link_libraries(unit_tests PRIVATE lob_core Catch2::Catch2WithMain)

# Register with CTest so `ctest` works out of the box
include(Catch2::Catch2WithMain)
include(CTest)
include(Catch)
catch_discover_tests(unit_tests)

# ── Python module (Phase 6 — placeholder) ───────────────────────────────────
# Uncomment when pybind11 bindings are ready:
# pybind11_add_module(lob_gym python/bindings/module.cpp)
# target_link_libraries(lob_gym PRIVATE lob_core)
