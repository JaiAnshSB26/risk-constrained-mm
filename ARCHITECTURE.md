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
│   │   ├── data/         # Data ingestion (Phase 4)
│   │   │   ├── tick.hpp             # Tick struct + TickAction enum
│   │   │   ├── market_data_parser.hpp # Zero-alloc CSV parser
│   │   │   └── replay_engine.hpp    # Historical tick replay
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

### 4.7 Market Data Parsing — Zero-Allocation I/O

The `MarketDataParser` ingests CSV tick data (Tardis.dev / Binance format)
with **zero per-row heap allocation**:

| Step              | Technique                                           |
|-------------------|-----------------------------------------------------|
| File loading      | Single `fread()` into `std::vector<char>` buffer    |
| Line splitting    | `string_view::find('\n')` + `remove_prefix()`      |
| Field extraction  | `string_view::find(',')` + `substr()` (zero-copy)   |
| Integer parsing   | `std::from_chars` (no locale, no allocation)         |
| Decimal parsing   | `std::from_chars` → `double`, then `llround(val / tick_size)` |

**CSV format** (first line = header, skipped):

```
timestamp,action,order_id,side,price,qty
1609459200000000,add,12345,bid,50000.50,1.500
```

**Why not `std::getline` / `std::stringstream`?**  Both allocate a new
`std::string` per call.  For million-row tick files, this produces millions
of heap allocations.  Our approach performs **zero** allocations during
parsing — only the output `std::vector<Tick>` allocates (once, via
`reserve()`).

Decimal-to-integer conversion uses `std::from_chars` (C++17) for the
floating-point parse, then `std::llround(value / tick_size)` to produce an
exact integer tick price.  The `ParseConfig` struct controls `tick_size` and
`lot_size` for price and quantity scaling.

### 4.8 Replay Engine

The `ReplayEngine` applies parsed ticks to a live `OrderBook`:

| Tick Action | OrderBook API Called                        |
|-------------|---------------------------------------------|
| `Add`       | `add_order(id, side, price, qty, ts)`       |
| `Cancel`    | `cancel_order(id)`                          |
| `Modify`    | `cancel_order(id)` + `add_order(id, ...)`   |
| `Trade`     | `place_order(id, price, qty, side, Market)`  |

**`replay(ticks)`** processes a batch and returns a `ReplayResult`
containing all generated trades, the count of processed ticks, and error
count.  **`replay_one(tick)`** is the single-step API for RL environment
interaction.

**`top_of_book()`** returns a `TopOfBook` snapshot (best\_bid, best\_ask,
timestamp) at any point during replay.

### 4.9 Compiler Discipline

| Flag                | Purpose                                              |
|---------------------|------------------------------------------------------|
| `-Werror`           | Zero warnings policy                                 |
| `-fno-exceptions`   | Exceptions banned from hot path (forces error codes)  |
| `-O3 -march=native` | Release-mode auto-vectorization + native ISA          |
| `-flto`             | Link-time optimization across translation units       |

Flags are applied only to project targets via `rcmm_set_strict_warnings()`,
not to third-party dependencies (Catch2, pybind11).

### 4.10 Hawkes Process Market Simulator

Phase 5 introduces a **1D Marked Hawkes Process** simulator for generating
realistic synthetic order-flow with self-exciting clustering — the hallmark
of high-frequency market microstructure data.

#### 4.10.1 Intensity Function

The conditional intensity of a 1D exponential-kernel Hawkes process is:

$$\lambda(t) \;=\; \mu \;+\; \sum_{t_i < t} \alpha \, e^{-\beta\,(t - t_i)}$$

| Parameter | Meaning                                |
|-----------|----------------------------------------|
| $\mu$     | Baseline (background) intensity        |
| $\alpha$  | Excitation jump per event              |
| $\beta$   | Exponential decay rate                 |

**Stationarity** requires the branching ratio $\alpha / \beta < 1$.
The theoretical steady-state expected intensity is
$E[\lambda] = \mu \, / \, (1 - \alpha/\beta)$.

#### 4.10.2 Regime Presets

| Regime           | $\mu$ | $\alpha$ | $\beta$ | Branching | $E[\lambda]$ |
|------------------|-------|----------|---------|-----------|---------------|
| `NORMAL_REGIME`  | 5.0   | 1.5      | 5.0     | 0.30      | ≈ 7.14        |
| `FLASH_CRASH_REGIME` | 5.0 | 9.5   | 10.0    | 0.95      | 100.0         |

Both are verified at compile time via `static_assert(is_stationary())`.

#### 4.10.3 Ogata's Modified Thinning Algorithm

The `HawkesSimulator` class implements **Ogata's modified thinning** to
sample exact event times without discretization:

1. Set upper-bound intensity $\bar{\lambda} = \lambda(t)$.
2. Draw candidate inter-arrival $u \sim \text{Exp}(\bar\lambda)$.
3. Advance time: $t \mathrel{+}= u$.  Decay:
   $\lambda(t) = \mu + (\lambda_{\text{prev}} - \mu)\,e^{-\beta\,u}$.
4. Accept with probability $\lambda(t)\,/\,\bar\lambda$ (thinning step).
5. On accept: emit event, jump $\lambda \mathrel{+}= \alpha$.
6. Repeat until the desired number of events is reached.

**RNG**: `std::mt19937_64` seeded at construction; `seed()` method for
reproducible experiments.  Pre-allocates output via `reserve(num_events)`.

#### 4.10.4 Mark Generation

Each accepted event is assigned a random **mark** (order-flow attributes)
controlled by the `MarkConfig` struct:

| Mark       | Distribution                                       |
|------------|----------------------------------------------------|
| Side       | Bernoulli with configurable `buy_prob` (default 0.5)|
| Price      | Uniform integer in `[mid - half_spread, mid + half_spread]` |
| Qty        | Uniform integer in `[min_qty, max_qty]`            |
| Action     | Categorical: `add_prob` / `cancel_prob` / `trade_prob` / remainder → Modify |
| `is_trade` | Derived: `action == TickAction::Trade`             |
| `order_id` | Sequential starting from 1                         |

Output is `std::vector<Tick>` — the same struct used by `ReplayEngine`,
ensuring full compatibility with the LOB pipeline.

#### 4.10.5 Test Coverage (23 test cases)

| Category   | Tests                                                          |
|------------|----------------------------------------------------------------|
| Params     | Stationarity, branching ratio, expected intensity, presets     |
| Ogata      | 10k-event runs for both regimes; empirical vs theoretical rate |
| Monotonicity | Strictly increasing timestamps; positive inter-arrival times |
| Marks      | Side/price/qty/action distributions within statistical bounds  |
| Regime     | Flash crash arrivals more clustered than normal                |
| Repro      | Same seed → identical output; reseed → different output        |
| Edge       | Small simulations (10 events); timestamp offset                |

### 4.11 Python Bindings & Gymnasium Environment (Phase 6)

Phase 6 bridges the C++ matching engine to Python for RL training via
pybind11, with a **strict zero-copy** observation strategy.

#### 4.11.1 Architecture Overview

```
Python (gymnasium.Env)           C++ (pybind11 module)
┌──────────────────────┐        ┌──────────────────────────────┐
│  LimitOrderBookEnv   │ ─────▶ │     MarketEnvironment        │
│                      │        │  ┌──────────┐ ┌───────────┐  │
│  obs_space (Box)     │        │  │OrderBook │ │ Hawkes    │  │
│  act_space (Box)     │        │  │(matching)│ │ Simulator │  │
│                      │        │  └──────────┘ └───────────┘  │
│  step(action)────────│───────▶│  step(bid_s, ask_s, ...)     │
│    → (obs, r, d, t, i)       │    → obs_buffer_ (zero-copy) │
└──────────────────────┘        └──────────────────────────────┘
```

#### 4.11.2 Zero-Copy Observation Buffer

The `MarketEnvironment` class holds a **pre-allocated `std::vector<double>`**
of size $4N + 2$ (where $N$ = `obs_depth`, default 5).  This buffer is
allocated once at construction and reused every `step()` call.

**Buffer layout** ($N = 5 \Rightarrow 22$ doubles):

| Index     | Content                                    |
|-----------|--------------------------------------------|
| 0–1       | Best bid: (normalised\_price, norm\_vol)   |
| 2–3       | 2nd bid level                              |
| …         | …                                          |
| 8–9       | 5th bid level                              |
| 10–11     | Best ask: (normalised\_price, norm\_vol)   |
| …         | …                                          |
| 18–19     | 5th ask level                              |
| 20        | Normalised inventory (inv / max\_inventory)|
| 21        | Normalised PnL (pnl / max\_pnl)           |

**pybind11 bridge**: The C++ pointer is exposed to Python as a
`pybind11::array_t<double>` with a no-op capsule destructor.  The NumPy
array returned to Python is a **view** — no `memcpy`, no heap allocation.

#### 4.11.3 Step Protocol

Each `step(bid_spread, ask_spread, bid_size, ask_size)` call:

1. **Cancel** the agent's previous bid/ask quotes.
2. **Place** new limit orders at `mid ± spread × tick_size`.
3. **Advance** the Hawkes simulation by `ticks_per_step` market events.
4. **Detect fills** on agent orders; update inventory and PnL.
5. **Fill** the observation buffer (zero allocation).
6. **Compute reward**: $\Delta(\text{mark-to-market}) - 0.01 \cdot \text{inventory}^2$.

Agent order IDs start at 10,000,000 to avoid collision with
simulator-generated IDs.

#### 4.11.4 Gymnasium Wrapper

`LimitOrderBookEnv(gymnasium.Env)` in `python/rcmm/env.py`:

| Property             | Value                                     |
|----------------------|-------------------------------------------|
| `observation_space`  | `Box(-inf, inf, shape=(22,), float64)`    |
| `action_space`       | `Box([1,1,1,1], [20,20,50,50], float64)` |
| `reset()`            | Returns `(obs, info)` per Gymnasium v1    |
| `step(action)`       | Returns `(obs, reward, term, trunc, info)`|

#### 4.11.5 Build & Module Delivery

CMake compiles `cpp/bindings.cpp` via `pybind11_add_module(_rcmm_core …)`,
producing `_rcmm_core.cp3XX-win_amd64.pyd` (or `.so` on Linux).  A
`POST_BUILD` custom command copies the artifact into `python/rcmm/` for
in-tree import.

On Windows with MinGW/UCRT64, `__init__.py` calls
`os.add_dll_directory(r"C:\msys64\ucrt64\bin")` to resolve the GCC runtime.

#### 4.11.6 Test Coverage (30 pytest cases)

| Category        | Tests                                                    |
|-----------------|----------------------------------------------------------|
| Import          | C++ module, env wrapper, package                         |
| Spaces          | obs shape/dtype, action shape/dtype/bounds               |
| Reset           | Return type, obs shape, info dict, finiteness            |
| Step            | 5-tuple, obs shape, reward type, terminated/truncated    |
| Episode         | Termination at max\_steps, reset-after-done              |
| Reproducibility | Same seed → identical trajectories                       |
| Zero-copy       | Stable data pointer across steps                         |
| Performance     | >10k steps/sec (measured: ~150k steps/sec)               |
| Gymnasium       | isinstance(env, gym.Env), space containment              |

### 4.12 Baseline RL Agent & Reward Shaping (Phase 7)

Phase 7 implements a clean, hackable PPO agent in PyTorch (CleanRL-style)
and shapes the reward function for inventory-averse market making.

#### 4.12.1 Reward Function

The reward function implements the standard inventory-averse market-making
objective:

$$R_t = \Delta \text{PnL}_t - \gamma \cdot (\text{Inventory}_t)^2$$

| Symbol | Meaning | Default |
|--------|---------|---------|
| $\Delta \text{PnL}_t$ | Change in mark-to-market PnL | — |
| $\gamma$ | Inventory aversion coefficient | 0.01 |
| $\text{Inventory}_t$ | Current net position (signed) | — |

The inventory aversion $\gamma$ is exposed as `EnvConfig.inventory_aversion`
in C++ and as a keyword argument `inventory_aversion` on the Python
`LimitOrderBookEnv` constructor.  Mark-to-market uses mid-price:
$\text{MtM}_t = \text{PnL}_{\text{cash}} + \text{Inventory}_t \times \text{mid}_t$.

#### 4.12.2 Actor-Critic Architecture

```
obs (22-d) ──► [Linear(22, 64) → Tanh
                Linear(64, 64)  → Tanh] ──► shared features (64-d)
                                                  │
                          ┌───────────────────────┤
                          ▼                       ▼
                   Actor head                Critic head
                 Linear(64, 4)             Linear(64, 1)
                   = mean(s)                  = V(s)
                 + log_std (4-d param)
```

| Component | Detail |
|-----------|--------|
| Shared encoder | 2× Linear(→64) + Tanh |
| Actor head | Linear(64, 4) = action mean |
| log\_std | Learnable parameter (state-independent, 4-d) |
| Critic head | Linear(64, 1) = state value V(s) |
| Total params | 5,961 |
| Initialisation | Orthogonal (gain=√2 encoder, 0.01 actor, 1.0 critic) |

**Policy distribution**: $\pi(a|s) = \mathcal{N}(\mu(s),\, \text{diag}(\exp(\log\sigma)))$

The 4 continuous action dimensions are:
$[bid\_spread,\, ask\_spread,\, bid\_size,\, ask\_size]$

#### 4.12.3 PPO Algorithm

| Step | Description |
|------|-------------|
| 1 | Collect rollout of $T$ transitions using current policy |
| 2 | Compute GAE advantages: $\hat{A}_t = \sum_{l=0}^{T-t-1} (\gamma\lambda)^l \delta_{t+l}$ |
| 3 | Shuffle into $K$ minibatches |
| 4 | For each epoch (default 10): clipped surrogate update |
| 5 | Repeat from step 1 |

**PPO clipped objective**:
$$L^{CLIP} = \hat{\mathbb{E}}_t \left[\min\left(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

| Hyperparameter | Symbol | Default |
|----------------|--------|---------|
| Discount | $\gamma$ | 0.99 |
| GAE λ | $\lambda$ | 0.95 |
| Clip ε | $\epsilon$ | 0.2 |
| Learning rate | $\eta$ | 3×10⁻⁴ |
| Entropy coef | $c_1$ | 0.01 |
| Value loss coef | $c_2$ | 0.5 |
| Max grad norm | — | 0.5 |
| Rollout steps | $T$ | 2048 |
| Minibatches | $K$ | 32 |
| Update epochs | — | 10 |

#### 4.12.4 Training Pipeline

The `train_baseline.py` script provides an end-to-end training loop:

1. Initialise `LimitOrderBookEnv` with Normal Hawkes regime.
2. Create `PPOAgent` with configurable hyperparameters.
3. Train for N timesteps (default 10,000 for smoke test).
4. Print per-update metrics (policy loss, value loss, entropy, KL, clip frac).
5. Verify no NaN and bounded memory growth via `tracemalloc`.

**Measured performance** (10k-step smoke test): ~895 steps/sec training
throughput (includes forward pass, env step, backward pass, and PPO update).
Memory growth: 81 KB over 10k steps (no leak).

#### 4.12.5 Test Coverage (28 pytest cases)

| Category         | Tests |
|------------------|-------|
| Reward function  | Quadratic penalty, γ configurable, zero-inventory, kwarg API |
| ActorCritic      | Output shapes, batch shapes, value output, gradient flow, action evaluation, orthogonal init, learnable log\_std |
| PPO agent        | Action bounds, rollout buffer fills, GAE finite, update metrics, param updates |
| Behavioral skew  | Extreme long (+100), extreme short (−100), all-zeros, large values |
| Training loop    | Short training (no crash/NaN), multi-episode rollouts |
| Rollout buffer   | Reset, manual GAE verification, terminal state handling |
| Determinism      | Same seed → same actions |
| PPO config       | Default values, minibatch size computation |

### 4.13 Risk-Constrained RL for Regime Shifts (Phase 8)

Phase 8 is the research core: introducing **non-stationarity** (regime shifts)
via domain randomization and modifying the PPO agent into a
**Risk-Constrained PPO (CPPO)** that optimises expected return while
explicitly penalising tail-risk via **Conditional Value at Risk (CVaR)**.

#### 4.13.1 Domain Randomization — Regime Shifts

On each `reset()`, the `RegimeRandomizationWrapper` (Gymnasium wrapper)
randomly samples Hawkes process parameters from one of two regime
distributions:

| Regime | μ | α | β | Branching Ratio | buy_prob |
|--------|---|---|---|-----------------|----------|
| **Normal** (80%) | N(5.0, 0.5) | N(1.5, 0.3) | N(5.0, 0.5) | ≈ 0.30 | 0.50 |
| **Flash Crash** (20%) | N(5.0, 0.5) | N(9.5, 0.5) | N(10.0, 0.5) | ≈ 0.95 | 0.15 |

All parameters are clipped to valid ranges and **stationarity is enforced**:
$\alpha < 0.98 \cdot \beta$ after sampling.

The wrapper reconstructs the C++ `MarketEnvironment` on each reset with the
sampled parameters, producing genuinely different market microstructure
per episode.  The `info` dict includes regime name and Hawkes parameters
for diagnostics.

**Implementation**: `python/rcmm/regime_wrapper.py`
- `RegimeSpec` dataclass — sampling distribution specification
- `RegimeRandomizationWrapper(gym.Wrapper)` — the Gymnasium wrapper
- Pre-defined: `NORMAL_REGIME_SPEC`, `FLASH_CRASH_REGIME_SPEC`

#### 4.13.2 CVaR — Conditional Value at Risk

CVaR (Expected Shortfall) quantifies the expected loss in the worst-α
fraction of outcomes:

$$\text{VaR}_\alpha = \inf\{ z : P(Z \le z) \ge \alpha \}$$
$$\text{CVaR}_\alpha = \mathbb{E}[Z \mid Z \le \text{VaR}_\alpha]$$

For a finite sample of rewards $Z = \{z_1, \ldots, z_n\}$:
1. Sort in ascending order
2. $k = \max(1, \lfloor n \cdot \alpha \rfloor)$
3. $\text{VaR}_\alpha = z_{(k)}$
4. $\text{CVaR}_\alpha = \frac{1}{|\{i : z_i \le \text{VaR}_\alpha\}|}
   \sum_{i:\, z_i \le \text{VaR}_\alpha} z_i$

**Properties verified by the test suite**:
- $\text{CVaR}_\alpha \le \text{VaR}_\alpha$ (tail is more extreme)
- $\text{CVaR}_\alpha \le \mathbb{E}[Z]$ (tail is worse than average)
- Negative skew → more negative CVaR

#### 4.13.3 CPPO — Constrained PPO Algorithm

The CPPO agent extends baseline PPO with a Lagrangian CVaR constraint:

$$\max_\theta\; \mathbb{E}[R(\tau)] \quad \text{s.t.}\; \text{CVaR}_\alpha(R) \ge d$$

Relaxed via Lagrangian:

$$L_{\text{CPPO}} = L_{\text{PPO}} + \lambda \cdot \max(0,\, d - \text{CVaR}_\alpha)$$

The Lagrangian multiplier $\lambda$ adapts via dual gradient ascent:

$$\lambda_{k+1} = \text{clamp}\!\left(\lambda_k + \eta_\lambda \cdot (d - \text{CVaR}_\alpha),\; 0,\; \lambda_{\max}\right)$$

| When... | Effect |
|---------|--------|
| Tail risk exceeds threshold ($\text{CVaR} < d$) | $\lambda$ increases → stronger penalty |
| Tail risk is acceptable ($\text{CVaR} \ge d$) | $\lambda$ decreases toward 0 |

| CPPO Hyperparameter | Symbol | Default |
|---------------------|--------|---------|
| CVaR confidence | $\alpha$ | 0.05 |
| CVaR threshold | $d$ | −10.0 |
| Lagrange learning rate | $\eta_\lambda$ | 0.01 |
| Initial $\lambda$ | — | 0.1 |
| Max $\lambda$ | $\lambda_{\max}$ | 10.0 |

**Implementation**: `python/rcmm/cppo.py`
- `compute_cvar(rewards, alpha)` — empirical CVaR computation
- `CPPOConfig(PPOConfig)` — extends PPO config with risk params
- `CPPOAgent` — full training loop with CVaR diagnostics

Network architecture is identical to baseline PPO (§4.12.2).

#### 4.13.4 C++ pybind11 Extensions

Phase 8 extends the pybind11 bindings to expose Hawkes parameters:

| Binding | Fields |
|---------|--------|
| `HawkesParams` | `mu`, `alpha`, `beta`, `is_stationary()`, `branching_ratio()`, `expected_intensity()`, `__repr__` |
| `MarkConfig` | `buy_prob`, `spread_mean`, `spread_std`, `qty_mean`, `qty_std`, `qty_lo`, `qty_hi` |
| `EnvConfig` | `hawkes_params` (readwrite), `mark_config` (readwrite) |

This allows Python-level regime randomization without C++ recompilation.

#### 4.13.5 Evaluation: PPO vs CPPO on Flash Crash

`python/scripts/evaluate_regimes.py` trains both agents on regime-randomized
environments and evaluates exclusively on Flash Crash conditions
(α=9.5, β=10.0, buy\_prob=0.15).

| Metric | Description |
|--------|-------------|
| Max Drawdown | $\max_t (\text{peak}_{0:t} - \text{PnL}_t)$ |
| Sharpe Ratio | $\frac{\mathbb{E}[R]}{\text{std}(R)}$ |
| CVaR | Bottom-20% average of episode returns |
| Avg Return | Mean episodic PnL |

**Expected result**: CPPO exhibits lower max drawdown and better CVaR
under flash crash, at the cost of slightly lower average return — a
classic risk-return trade-off.

#### 4.13.6 Test Coverage (37 pytest cases)

| Category | Tests |
|----------|-------|
| CVaR math | Known arrays (5%, 20%, 50%), all-same, single, CVaR≤VaR, CVaR≤mean, negative skew |
| Domain randomization | Regime info, param variation, stationarity, distribution, sell pressure, step-after-reset, custom spec, seed reproducibility |
| CPPO agent | Action validity, gradient flow, CVaR metrics, λ increase on violation, λ stays 0 on no violation, param updates |
| CPPO training | Short loop (no NaN), regime-wrapped training, Lagrange trajectory |
| Eval helpers | Max drawdown (known, monotonic up/down, empty), Sharpe (positive, zero-std, single) |
| Bindings | HawkesParams roundtrip, EnvConfig integration, MarkConfig, __repr__ |
| Config | CPPOConfig inherits PPO defaults |

### 4.14 Evaluation & Diebold-Mariano Rigor (Phase 9)

Phase 9 proves statistical significance by pitting the CPPO agent against
a classical mathematical baseline on out-of-sample data and applying the
Diebold-Mariano test to the PnL differentials.

#### 4.14.1 Avellaneda-Stoikov Baseline Agent

A deterministic, non-learning market-making strategy based on the
Avellaneda-Stoikov (2008) model.  No neural networks — just closed-form
reservation-price mathematics.

**Reservation price** (inventory-adjusted mid):

$$r(s, q) = s - q \cdot \gamma \cdot \sigma^2$$

**Optimal spread**:

$$\delta = \gamma \cdot \sigma^2 + \frac{2}{\gamma} \cdot \ln\!\left(1 + \frac{\gamma}{\kappa}\right)$$

**Quotes**:

$$\text{bid} = r - \delta/2 \qquad \text{ask} = r + \delta/2$$

| Symbol | Meaning | Default |
|--------|---------|---------|
| $s$ | Mid-price | from obs |
| $q$ | Inventory (signed) | from obs |
| $\gamma$ | Risk aversion | 0.1 |
| $\sigma$ | Volatility estimate | 2.0 |
| $\kappa$ | Order arrival intensity | 1.5 |

**Key properties**:
- Zero inventory → symmetric spread (bid = ask distance from mid).
- Long inventory → wider bid / tighter ask (skew to sell).
- Short inventory → tighter bid / wider ask (skew to buy).

**Implementation**: `python/rcmm/baselines.py`
- `AvellanedaStoikovConfig` — dataclass with all parameters
- `AvellanedaStoikovAgent` — `select_action(obs)` matching PPO/CPPO interface

Reference: Avellaneda, M. & Stoikov, S. (2008).
"High-Frequency Trading in a Limit Order Book."
*Quantitative Finance*, 8(3), 217–224.

#### 4.14.2 Diebold-Mariano Test

The DM test (Diebold & Mariano, 1995) tests whether two forecasting /
trading strategies have equal predictive accuracy.

Let $d_t = r_{CPPO,t} - r_{Baseline,t}$ be the return differential at step $t$.

$$H_0: \mathbb{E}[d_t] = 0 \qquad \text{(no performance difference)}$$

**DM statistic**:

$$DM = \frac{\bar{d}}{\sqrt{\hat{V}(\bar{d})}}$$

**Newey-West HAC variance** (accounts for autocorrelation):

$$\hat{V}(\bar{d}) = \frac{1}{T}\left[\hat{\gamma}(0) + 2\sum_{\tau=1}^{h}\hat{\gamma}(\tau)\right]$$

where $\hat{\gamma}(\tau) = \frac{1}{T}\sum_{t=|\tau|+1}^{T}(d_t - \bar{d})(d_{t-|\tau|} - \bar{d})$

| Parameter | Value |
|-----------|-------|
| Truncation lag $h$ | $\lfloor T^{1/3} \rfloor$ (default) |
| Distribution | $DM \xrightarrow{d} \mathcal{N}(0,1)$ |
| Two-sided p-value | $p = 2\Phi(-|DM|)$ |

**Robustness**: The implementation handles degenerate cases (zero variance,
constant differentials, very short series) without division-by-zero errors.

**Implementation**: `python/rcmm/stats.py`
- `diebold_mariano(pnl_1, pnl_2, h=None, one_sided=False) → (dm_stat, p_value)`

Reference: Diebold, F.X. & Mariano, R.S. (1995).
"Comparing Predictive Accuracy."
*Journal of Business & Economic Statistics*, 13(3), 253–263.

#### 4.14.3 Out-of-Sample Evaluation Pipeline

`python/scripts/evaluate_oos.py` implements the full OOS evaluation:

| Step | Description |
|------|-------------|
| 1 | Generate synthetic OOS CSV (10,000 ticks) via `generate_oos_csv()` |
| 2 | Train CPPO agent on regime-randomised environment |
| 3 | Create OOS environment (Flash Crash regime: α=9.0, β=10.0, buy_prob=0.2) |
| 4 | Run CPPO and AS baseline on same OOS environment |
| 5 | Collect step-by-step PnL for both agents |
| 6 | Apply Diebold-Mariano test to PnL differentials |
| 7 | Export trajectories to `results/oos_trajectories.csv` (pandas DataFrame) |

**CSV output columns**: step, cppo\_reward, baseline\_reward,
cppo\_cumulative\_pnl, baseline\_cumulative\_pnl, differential.

#### 4.14.4 Test Coverage (37 pytest cases)

| Category | Tests |
|----------|-------|
| AS interface | Action shape, log\_prob/value tensors, finiteness (5) |
| AS symmetry | Zero-inventory symmetric spreads, positive spread (2) |
| AS inventory skew | Long/short wider bid/ask, larger inventory → larger skew, get\_spreads (6) |
| AS bounds | Extreme +/− inventory clamped, fixed order size (3) |
| AS determinism | Same obs → same action (1) |
| AS config | γ wider spread, σ wider spread, default values (3) |
| AS integration | Full episode through environment (1) |
| DM core | Clear outperformance p<0.05, underperformance, identical, no-diff (4) |
| DM symmetry | Swap negates statistic, equal p-values (1) |
| DM edge cases | Too-short raises, different-lengths raises, T=2, constant, zero-diff, no-ZeroDivision (6) |
| DM one-sided | Positive direction, wrong direction (2) |
| DM HAC variance | Custom lag, large lag (2) |
| DM math validation | Manual hand-computed reference (1) |

---

## 5. Phase Tracker

| Phase | Description                          | Status      |
|-------|--------------------------------------|-------------|
| 1     | Foundation, Build System & Git Init  | **Complete** |
| 2     | High-Performance LOB API & O(1) Lookup | **Complete** |
| 3     | Matching Engine Core                   | **Complete** |
| 4     | Crypto Data Ingestion & Replay       | **Complete** |
| 5     | Hawkes Process Market Simulator      | **Complete** |
| 6     | Python Bindings & Gymnasium Env      | **Complete** |
| 7     | Baseline RL Agent & Reward Shaping   | **Complete** |
| 8     | Risk-Constrained RL (Research Core)  | **Complete** |
| 9     | Evaluation & Diebold-Mariano Rigor   | **Complete** |
| 10    | Publication Artifacts & V1.0 Release | Planned     |

---

## 6. Conventions

- **Namespace**: All C++ code lives in `namespace rcmm`.
- **Include style**: `#include "lob/types.hpp"` — quoted, path from `cpp/include/`.
- **Commit style**: [Conventional Commits](https://www.conventionalcommits.org/) —
  `feat(scope):`, `fix(scope):`, `test:`, `docs:`, `ci:`, `chore:`.
- **Branch model**: Trunk-based on `main`; feature work in `phase-N` branches
  if needed.
