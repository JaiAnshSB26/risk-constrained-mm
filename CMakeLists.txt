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

# Aggressive warning flags — applied only to OUR targets (not fetched deps).
# We use a function to apply them so they don't leak to Catch2/pybind11.
function(rcmm_set_strict_warnings target)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE
            -Wall -Wextra -Wpedantic -Werror
            -Wconversion -Wsign-conversion -Wshadow
            -Wnon-virtual-dtor -Wold-style-cast -Wcast-align
            -Woverloaded-virtual -Wformat=2
            -fno-exceptions    # hot-path must never throw
        )
        if(CMAKE_BUILD_TYPE STREQUAL "Release")
            target_compile_options(${target} PRIVATE -O3 -march=native -flto)
            target_link_options(${target} PRIVATE -flto)
        endif()
    elseif(MSVC)
        target_compile_options(${target} PRIVATE /W4 /WX /permissive- /Zc:__cplusplus)
        if(CMAKE_BUILD_TYPE STREQUAL "Release")
            target_compile_options(${target} PRIVATE /O2 /GL)
            target_link_options(${target} PRIVATE /LTCG)
        endif()
    endif()
endfunction()

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
rcmm_set_strict_warnings(lob_core)

# ── Testing ──────────────────────────────────────────────────────────────────
enable_testing()

file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS "tests/cpp/*.cpp")

add_executable(unit_tests ${TEST_SOURCES})
target_link_libraries(unit_tests PRIVATE lob_core Catch2::Catch2WithMain)
rcmm_set_strict_warnings(unit_tests)

# Register with CTest so `ctest` works out of the box
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)
catch_discover_tests(unit_tests)

# ── Python module (Phase 6) ──────────────────────────────────────────────────
# pybind11 extension module: _rcmm_core.pyd / _rcmm_core.so
# This is the zero-copy bridge between the C++ matching engine and Python.
pybind11_add_module(_rcmm_core cpp/bindings.cpp)
target_link_libraries(_rcmm_core PRIVATE lob_core)
target_include_directories(_rcmm_core PRIVATE cpp/include)

# Apply strict warnings to the bindings too, but we must allow exceptions
# for pybind11 (it uses them internally for type conversion errors).
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(_rcmm_core PRIVATE
        -Wall -Wextra -Wpedantic -Werror
        -Wconversion -Wsign-conversion -Wshadow
        -Wnon-virtual-dtor -Wcast-align
        -Woverloaded-virtual -Wformat=2
        # NOTE: no -fno-exceptions — pybind11 needs exceptions
        # NOTE: no -Wold-style-cast — pybind11 macros use C-style casts
    )
endif()

# Copy the built module to python/rcmm/ so it's importable in-tree.
add_custom_command(TARGET _rcmm_core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:_rcmm_core>
        ${CMAKE_SOURCE_DIR}/python/rcmm/$<TARGET_FILE_NAME:_rcmm_core>
    COMMENT "Copying _rcmm_core module to python/rcmm/"
)
