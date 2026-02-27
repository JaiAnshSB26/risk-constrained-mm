# ARCHITECTURE.md

> **risk-constrained-mm** — A high-frequency, research-grade C++ Limit Order Book
> matching engine with a risk-constrained Reinforcement Learning market-making
> agent.  Built for publication at a top ML/Finance conference (ICAIF / NeurIPS
> workshops).

---

## 1. Project Philosophy: "One Core, Two Outputs"

A single, highly deterministic, low-latency **C++20 Limit Order Book (LOB)**
matching engine serves two roles:

1. **Systems-engineering artifact** — demonstrating C++ proficiency with
   zero-allocation hot paths, cache-aware data structures, and strict
   `-Werror -fno-exceptions` discipline.
2. **Simulated Gymnasium environment** — training a **Risk-Constrained
   Reinforcement Learning (RL)** market-making agent that must survive
   non-stationary market regimes (volatility spikes, flash crashes).

---

## 2. Directory Layout

```
risk-constrained-mm/
├── cpp/
│   ├── include/
│   │   ├── lob/          # Core LOB data structures
│   │   │   ├── types.hpp        # Fixed-width types, enums, constants
│   │   │   ├── order.hpp        # Order struct with intrusive list ptrs
│   │   │   ├── pool.hpp         # Zero-allocation OrderPool (free-list)
│   │   │   ├── order_queue.hpp  # Intrusive doubly-linked list (price level)
│   │   │   ├── price_level.hpp  # Price → OrderQueue binding
│   │   │   ├── order_map.hpp    # O(1) OrderId → Order* hash map
│   │   │   └── order_book.hpp   # Central LOB with flat price-level arrays
│   │   ├── engine/       # Matching engine (Phase 3)
│   │   ├── sim/          # Hawkes process simulator (Phase 5)
│   │   └── data/         # Tardis.dev / Binance parser (Phase 4)
│   ├── src/              # .cpp translation units
│   └── bench/            # Microbenchmarks
├── python/
│   ├── env/              # Gymnasium environment wrapper (Phase 6)
│   └── bindings/         # pybind11 C++ ↔ Python bridge (Phase 6)
├── rl/
│   ├── agents/           # PPO / CPPO agent implementations (Phases 7–8)
│   ├── configs/          # Hyperparameter YAML files
│   └── training/         # Training loop, logging, checkpointing
├── data/
│   ├── raw/              # Downloaded Tardis.dev / Binance data (gitignored)
│   └── processed/        # Pre-processed data ready for replay
├── tests/
│   ├── cpp/              # Catch2 C++ unit tests
│   └── python/           # pytest Python tests
├── scripts/              # Build / data-download helpers
├── docs/                 # Extended documentation, paper drafts
├── .github/workflows/    # CI pipeline
├── CMakeLists.txt        # Root build configuration
└── pyproject.toml        # Python project metadata & tool config
```

---

## 3. Tech Stack

| Layer           | Technology                | Rationale                                     |
|-----------------|---------------------------|-----------------------------------------------|
| Core engine     | **C++20**                 | `constexpr`, concepts, ranges, `std::span`    |
| Build system    | **CMake 3.20+**           | FetchContent for Catch2 & pybind11            |
| C++ tests       | **Catch2 v3**             | Modern, header-friendly, `catch_discover_tests`|
| Python bridge   | **pybind11 v2.12**        | Zero-copy NumPy buffer protocol               |
| RL framework    | **PyTorch 2.x**           | Ubiquitous RL ecosystem                       |
| Gym interface   | **Gymnasium 0.29+**       | Standard RL environment API                   |
| Python tests    | **pytest**                | Fixtures, parametrize, benchmarks             |
| Linting         | **ruff + mypy**           | Fast, strict Python quality                   |
| CI              | **GitHub Actions**        | GCC 13 + Clang 17 matrix / Python 3.12        |
| Data source     | **Tardis.dev / Binance**  | L2 depth + L3 trade tick data for BTC/USDT     |

---

## 4. Core C++ Design Decisions

### 4.1 Zero-Allocation Guarantee

The matching engine's hot path performs **no heap allocation** after
initialization:

| Component     | Storage Strategy                                    |
|---------------|-----------------------------------------------------|
| `OrderPool`   | `std::array<Order, N>` with free-list (LIFO)        |
| `OrderQueue`  | Intrusive doubly-linked list (ptrs inside `Order`)   |
| `PriceLevel`  | `unique_ptr<PriceLevel[]>` — allocated once at init  |
| `OrderMap`    | Open-addressing hash table — allocated once at init  |
| `OrderBook`   | Flat contiguous arrays per side, O(1) price lookup   |

**Why no `std::map`?**  `std::map` performs one `new` per insert (red-black
tree nodes are heap-allocated).  A flat array indexed by price offset gives
O(1) lookups with perfect spatial locality and zero allocation.

### 4.2 Order Struct Layout

```cpp
struct Order {
    OrderId   id;          // 8 bytes
    Symbol    symbol;      // 4 bytes
    Side      side;        // 1 byte
    OrderType type;        // 1 byte
    OrderStatus status;    // 1 byte
    // 1 byte padding
    Price     price;       // 8 bytes
    Qty       qty;         // 8 bytes
    Qty       filled;      // 8 bytes
    Sequence  seq;         // 8 bytes
    Timestamp timestamp;   // 8 bytes
    Order*    prev;        // 8 bytes  ← intrusive list
    Order*    next;        // 8 bytes  ← intrusive list
};
static_assert(std::is_trivially_copyable_v<Order>);
```

Hot fields (price, qty, pointers) are packed near each other to maximize
cache-line utilization during matching.

### 4.3 OrderMap — O(1) Order Lookup

The `OrderMap` provides constant-time `OrderId → Order*` lookup, insert, and
delete with zero hot-path allocation:

| Property              | Detail                                            |
|-----------------------|---------------------------------------------------|
| Hash algorithm        | **splitmix64 finalizer** (excellent avalanche)     |
| Collision strategy    | **Linear probing** with power-of-2 table           |
| Deletion strategy     | **Backward-shift** — no tombstones, keeps chains   |
| Load factor           | ≤ 50 % (table sized ≥ 2× `PoolCap`)               |
| Capacity              | Rounded to next power-of-2 ≥ 2 × `max_elements`   |
| Memory                | Single `unique_ptr<Slot[]>` allocated at init      |

**Why not `std::unordered_map`?**  It performs one `new` per bucket/node
insertion and has unpredictable rehash stalls.  Our pre-sized open-addressing
table guarantees no allocation after construction and achieves excellent
cache locality for sequential probes.

### 4.4 Add / Cancel API

```
add_order(id, side, price, qty, ts)  →  Order*   (nullptr on failure)
cancel_order(id)                     →  bool     (false if not found)
find_order(id)                       →  Order*   (nullptr if not found)
```

**`add_order` hot path** (zero allocation):
1. `pool_.allocate()` — O(1) free-list pop
2. Set order fields + assign monotonic sequence number
3. `order_map_.insert(id, ptr)` — O(1) amortized hash insert
4. `level.queue.push_back(ptr)` — O(1) intrusive list append
5. Update `best_bid_` / `best_ask_` — O(1) compare

**`cancel_order` hot path**:
1. `order_map_.find(id)` — O(1) hash lookup
2. `level.queue.remove(ptr)` — O(1) intrusive unlink
3. `order_map_.erase(id)` — O(1) backward-shift delete
4. `pool_.deallocate(ptr)` — O(1) free-list push
5. If level is now empty and was best, **scan** for new best — O(levels) worst
   case, O(1) amortized

### 4.5 Price Representation

Prices are stored as **`int64_t` tick units**, not floating-point:

- Eliminates rounding drift.
- Integer compare/arithmetic is faster and deterministic.
- For BTC/USDT with tick = 0.01 USDT, price 65432.10 → `6'543'210`.

### 4.6 Matching Engine — Price-Time Priority

The `place_order()` API is the single entry-point for aggressive order flow.
It performs crossing (matching) before optionally resting the remainder.

```
place_order(id, price, qty, side, type, ts)  →  std::vector<Trade>
```

**Matching loop** (Price priority × Time priority):

```
1. While remaining_qty > 0 AND opposite side has resting orders:
     a. Check price limit (Limit orders only).
     b. Walk the best-price level’s OrderQueue from HEAD (oldest = time prio).
     c. For each maker order:
        • fill = min(remaining, maker.leaves())
        • Emit Trade{maker_id, taker_id, price, fill}.
        • If maker fully filled → remove from queue, erase from map,
          return to pool.
        • If partially filled → update maker.filled, set PartialFill status.
     d. If level is empty → scan_best_ask / scan_best_bid for next level.
2. If Limit order AND remaining > 0 → add_order() to rest on the book.
3. Return trades vector.
```

**Order type behaviour:**

| Type   | Matches at | Rests remainder? |
|--------|---------------------------------------------|------------------|
| Limit  | Opposite levels with favorable price | Yes |
| Market | All opposite levels regardless of price | Never |

**Trade struct:**

```cpp
struct Trade {
    OrderId   maker_id;    // resting order that was hit
    OrderId   taker_id;    // aggressive incoming order
    Price     price;       // execution price (always maker’s level)
    Qty       qty;         // filled quantity
    Side      taker_side;  // direction of the aggressor
    Timestamp timestamp;   // event time
};
static_assert(std::is_trivially_copyable_v<Trade>);
```

The trades vector is `reserve(64)`’d to minimise reallocation on the
Python-bridge path.  On the hot C++ path, a pre-allocated ring buffer
can be substituted in Phase 6.

### 4.7 Compiler Discipline

| Flag                | Purpose                                              |
|---------------------|------------------------------------------------------|
| `-Werror`           | Zero warnings policy                                 |
| `-fno-exceptions`   | Exceptions banned from hot path (forces error codes)  |
| `-O3 -march=native` | Release-mode auto-vectorization + native ISA          |
| `-flto`             | Link-time optimization across translation units       |

Flags are applied only to project targets via `rcmm_set_strict_warnings()`,
not to third-party dependencies (Catch2, pybind11).

---

## 5. Phase Tracker

| Phase | Description                          | Status      |
|-------|--------------------------------------|-------------|
| 1     | Foundation, Build System & Git Init  | **Complete** |
| 2     | High-Performance LOB API & O(1) Lookup | **Complete** |
| 3     | Matching Engine Core                   | **Complete** |
| 4     | Crypto Data Ingestion & Replay       | Planned     |
| 5     | Hawkes Process Market Simulator      | Planned     |
| 6     | Python Bindings & Gymnasium Env      | Planned     |
| 7     | Baseline RL Agent & Reward Shaping   | Planned     |
| 8     | Risk-Constrained RL (Research Core)  | Planned     |
| 9     | Evaluation & Diebold-Mariano Rigor   | Planned     |
| 10    | Publication Artifacts & V1.0 Release | Planned     |

---

## 6. Conventions

- **Namespace**: All C++ code lives in `namespace rcmm`.
- **Include style**: `#include "lob/types.hpp"` — quoted, path from `cpp/include/`.
- **Commit style**: [Conventional Commits](https://www.conventionalcommits.org/) —
  `feat(scope):`, `fix(scope):`, `test:`, `docs:`, `ci:`, `chore:`.
- **Branch model**: Trunk-based on `main`; feature work in `phase-N` branches
  if needed.
