// ============================================================================
// risk-constrained-mm :: cpp/bindings.cpp
// ============================================================================
// pybind11 module exposing MarketEnvironment to Python.
//
// Zero-copy strategy:
//   The MarketEnvironment holds a pre-allocated flat std::vector<double>
//   observation buffer.  We expose it to Python as a NumPy array via
//   pybind11::array_t<double> with a base-object prevent-dealloc guard.
//   The buffer pointer is stable across step() calls — no allocation.
//
// Compiled as: _rcmm_core.pyd / _rcmm_core.so
// ============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "env/market_env.hpp"

namespace py = pybind11;

// ── Zero-copy wrapper ───────────────────────────────────────────────────────
// Returns a NumPy array that views the environment's internal obs buffer
// without any copy.  The capsule prevents NumPy from trying to free the
// pointer (the MarketEnvironment owns the memory).

static py::array_t<double> make_obs_view(rcmm::MarketEnvironment& env) {
    auto* ptr = const_cast<double*>(env.obs_data());
    auto  size = static_cast<py::ssize_t>(env.obs_size());

    // Create a capsule that references the env (prevents dangling pointer).
    // The capsule destructor is a no-op because `env` owns the buffer.
    py::capsule dummy(ptr, [](void*) {});

    return py::array_t<double>(
        {size},          // shape
        {sizeof(double)}, // strides
        ptr,              // data pointer
        dummy             // base object (prevents dealloc)
    );
}

// ── Module definition ───────────────────────────────────────────────────────

PYBIND11_MODULE(_rcmm_core, m) {
    m.doc() = "C++ core for the risk-constrained market-making environment";

    // ── HawkesParams ────────────────────────────────────────────────────────
    py::class_<rcmm::HawkesParams>(m, "HawkesParams")
        .def(py::init<>())
        .def_readwrite("mu",    &rcmm::HawkesParams::mu)
        .def_readwrite("alpha", &rcmm::HawkesParams::alpha)
        .def_readwrite("beta",  &rcmm::HawkesParams::beta)
        .def("is_stationary",   &rcmm::HawkesParams::is_stationary)
        .def("branching_ratio", &rcmm::HawkesParams::branching_ratio)
        .def("expected_intensity", &rcmm::HawkesParams::expected_intensity)
        .def("__repr__", [](const rcmm::HawkesParams& p) {
            return "HawkesParams(mu=" + std::to_string(p.mu)
                 + ", alpha=" + std::to_string(p.alpha)
                 + ", beta=" + std::to_string(p.beta) + ")";
        });

    // ── MarkConfig ──────────────────────────────────────────────────────────
    py::class_<rcmm::MarkConfig>(m, "MarkConfig")
        .def(py::init<>())
        .def_readwrite("mid_price",    &rcmm::MarkConfig::mid_price)
        .def_readwrite("half_spread",  &rcmm::MarkConfig::half_spread)
        .def_readwrite("min_qty",      &rcmm::MarkConfig::min_qty)
        .def_readwrite("max_qty",      &rcmm::MarkConfig::max_qty)
        .def_readwrite("buy_prob",     &rcmm::MarkConfig::buy_prob)
        .def_readwrite("add_prob",     &rcmm::MarkConfig::add_prob)
        .def_readwrite("cancel_prob",  &rcmm::MarkConfig::cancel_prob)
        .def_readwrite("trade_prob",   &rcmm::MarkConfig::trade_prob);

    // ── EnvConfig ───────────────────────────────────────────────────────────
    py::class_<rcmm::EnvConfig>(m, "EnvConfig")
        .def(py::init<>())
        .def_readwrite("tick_size",      &rcmm::EnvConfig::tick_size)
        .def_readwrite("base_price",     &rcmm::EnvConfig::base_price)
        .def_readwrite("num_levels",     &rcmm::EnvConfig::num_levels)
        .def_readwrite("obs_depth",      &rcmm::EnvConfig::obs_depth)
        .def_readwrite("ticks_per_step", &rcmm::EnvConfig::ticks_per_step)
        .def_readwrite("max_steps",      &rcmm::EnvConfig::max_steps)
        .def_readwrite("warmup_ticks",   &rcmm::EnvConfig::warmup_ticks)
        .def_readwrite("max_inventory",  &rcmm::EnvConfig::max_inventory)
        .def_readwrite("max_pnl",        &rcmm::EnvConfig::max_pnl)
        .def_readwrite("max_volume",     &rcmm::EnvConfig::max_volume)
        .def_readwrite("inventory_aversion", &rcmm::EnvConfig::inventory_aversion)
        .def_readwrite("hawkes_params",  &rcmm::EnvConfig::hawkes_params)
        .def_readwrite("mark_config",    &rcmm::EnvConfig::mark_config)
        .def_readwrite("seed",           &rcmm::EnvConfig::seed);

    // ── StepResult ──────────────────────────────────────────────────────────
    py::class_<rcmm::StepResult>(m, "StepResult")
        .def_readonly("reward",    &rcmm::StepResult::reward)
        .def_readonly("done",      &rcmm::StepResult::done)
        .def_readonly("inventory", &rcmm::StepResult::inventory)
        .def_readonly("pnl",       &rcmm::StepResult::pnl)
        .def_readonly("step_num",  &rcmm::StepResult::step_num)
        .def_readonly("fills",     &rcmm::StepResult::fills);

    // ── MarketEnvironment ───────────────────────────────────────────────────
    py::class_<rcmm::MarketEnvironment>(m, "MarketEnvironment")
        .def(py::init<rcmm::EnvConfig>(), py::arg("config") = rcmm::EnvConfig{})

        .def("obs_size", &rcmm::MarketEnvironment::obs_size)

        .def("reset", [](rcmm::MarketEnvironment& self) {
            self.reset();
            return make_obs_view(self);
        }, "Reset the environment and return zeroed observation (zero-copy).")

        .def("step", [](rcmm::MarketEnvironment& self,
                        double bid_spread, double ask_spread,
                        double bid_size,   double ask_size) {
            auto result = self.step(bid_spread, ask_spread, bid_size, ask_size);
            py::dict info;
            info["inventory"] = result.inventory;
            info["pnl"]       = result.pnl;
            info["step_num"]  = result.step_num;
            info["fills"]     = result.fills;
            return py::make_tuple(
                make_obs_view(self),   // obs  (zero-copy)
                result.reward,         // reward
                result.done,           // terminated
                false,                 // truncated
                info                   // info dict
            );
        }, py::arg("bid_spread"), py::arg("ask_spread"),
           py::arg("bid_size"),   py::arg("ask_size"),
           "Step the environment: returns (obs, reward, terminated, truncated, info).")

        .def_property_readonly("inventory", &rcmm::MarketEnvironment::inventory)
        .def_property_readonly("pnl",       &rcmm::MarketEnvironment::pnl)
        .def_property_readonly("step_count", &rcmm::MarketEnvironment::step_count)
        .def_property_readonly("best_bid",  &rcmm::MarketEnvironment::best_bid)
        .def_property_readonly("best_ask",  &rcmm::MarketEnvironment::best_ask);
}
