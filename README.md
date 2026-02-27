# risk-constrained-mm

**Risk-Constrained Reinforcement Learning for Market Making under Non-Stationary Regimes**

A high-frequency, research-grade C++20 Limit Order Book matching engine with
zero-copy Python bindings, powering a CVaR-constrained PPO agent trained on
Hawkes-process regime shifts and real Binance BTC/USDT data.

> _One Core, Two Outputs_ — a systems-engineering artifact **and** a
> conference-ready RL research environment.

---

## Quick Start

### Prerequisites

- **CMake** ≥ 3.20
- **C++20 compiler** (GCC 13+, Clang 17+, or MSVC 2022+)
- **Python** ≥ 3.10
- **Git**

### Build the C++ engine & run tests

```bash
# Clone
git clone https://github.com/<your-org>/risk-constrained-mm.git
cd risk-constrained-mm

# Build (CMake fetches Catch2 + pybind11 automatically)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run C++ tests
cd build && ctest --output-on-failure -j$(nproc)
```

### Run Python tests

```bash
pip install numpy gymnasium pytest
python -m pytest tests/python/ -v
```

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design decisions,
data-structure rationale, and the 10-phase development roadmap.

---

## Project Status

| Phase | Description                              | Status       |
|-------|------------------------------------------|--------------|
| 1     | Foundation, Build System & Git Init      | **Complete** |
| 2     | High-Performance C++ LOB Data Structures | Planned      |
| 3     | Matching Engine Core                     | Planned      |
| 4     | Crypto Data Ingestion & Replay           | Planned      |
| 5     | Hawkes Process Market Simulator          | Planned      |
| 6     | Python Bindings & Gymnasium Env          | Planned      |
| 7     | Baseline RL Agent & Reward Shaping       | Planned      |
| 8     | Risk-Constrained RL (Research Core)      | Planned      |
| 9     | Evaluation & Diebold-Mariano Rigor       | Planned      |
| 10    | Publication Artifacts & V1.0 Release     | Planned      |

---

## License

MIT