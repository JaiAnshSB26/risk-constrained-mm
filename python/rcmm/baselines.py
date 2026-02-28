# ============================================================================
# risk-constrained-mm :: python/rcmm/baselines.py
# ============================================================================
"""
Classical market-making baselines for ablation against RL agents.

Implements the **Avellaneda-Stoikov (AS)** symmetric quoting strategy
as a deterministic, non-learning baseline.  No neural networks —
just closed-form reservation-price mathematics.

Mathematical formulation (simplified Avellaneda-Stoikov 2008):

    Reservation price:
        r(s, q, t) = s - q · γ · σ²

    Optimal spread:
        δ = γ · σ² + (2/γ) · ln(1 + γ/κ)

    Quotes:
        bid = r - δ/2
        ask = r + δ/2

where:
    s  = mid-price
    q  = inventory (signed: +long, −short)
    γ  = risk aversion parameter
    σ  = volatility estimate (simplified: fixed or rolling)
    κ  = order arrival intensity parameter

All baselines implement ``select_action(obs) → (action, log_prob, value)``
matching the PPO/CPPO agent interface for drop-in evaluation.

Reference:
    Avellaneda, M. & Stoikov, S. (2008).
    "High-Frequency Trading in a Limit Order Book."
    Quantitative Finance, 8(3), 217–224.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import torch


@dataclasses.dataclass
class AvellanedaStoikovConfig:
    """Configuration for the Avellaneda-Stoikov baseline agent."""

    # Risk aversion — higher γ → wider spreads, faster inventory decay.
    gamma: float = 0.1

    # Volatility estimate (σ) — fixed for simplicity (tick units per step).
    sigma: float = 2.0

    # Order arrival intensity parameter κ (higher → tighter spreads).
    kappa: float = 1.5

    # Observation layout: which indices hold inventory & mid-price info.
    # Default: obs[-2] = normalised_inventory, obs[-1] = normalised_pnl.
    inventory_index: int = -2

    # Normalisation.
    max_inventory: float = 100.0

    # Clamp bounds for the output action [bid_spread, ask_spread, bid_size, ask_size].
    min_spread: float = 1.0
    max_spread: float = 20.0
    order_size: float = 5.0     # fixed order size (lots)


class AvellanedaStoikovAgent:
    """
    Avellaneda-Stoikov market-making agent (deterministic baseline).

    Implements the same ``select_action(obs)`` interface as PPO/CPPO agents
    so it can be used interchangeably in evaluation scripts.

    The agent computes:
        1. Reservation price offset from mid (inventory skew).
        2. Optimal spread from risk-aversion and volatility.
        3. Bid/ask quotes as [bid_spread, ask_spread, bid_size, ask_size].

    Since this is deterministic, log_prob and value are dummy tensors.
    """

    def __init__(self, cfg: AvellanedaStoikovConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else AvellanedaStoikovConfig()

    def select_action(
        self, obs: np.ndarray
    ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """
        Compute AS quotes from the observation vector.

        Args:
            obs: Flat observation array from the environment.
                 obs[inventory_index] = normalised inventory.

        Returns:
            action: [bid_spread, ask_spread, bid_size, ask_size]
            log_prob: dummy scalar tensor (0.0)
            value: dummy scalar tensor (0.0)
        """
        cfg = self.cfg

        # Extract normalised inventory and de-normalise.
        norm_inv = float(obs[cfg.inventory_index])
        inventory = norm_inv * cfg.max_inventory

        # ── Reservation price offset ─────────────────────────────────────
        # r = mid - q · γ · σ²
        # The offset from mid in tick-units:
        #   reservation_offset = -q · γ · σ²
        # Positive offset → reservation price below mid (long inventory skew).
        reservation_offset = -inventory * cfg.gamma * (cfg.sigma ** 2)

        # ── Optimal spread ───────────────────────────────────────────────
        # δ = γ · σ² + (2/γ) · ln(1 + γ/κ)
        spread = (
            cfg.gamma * (cfg.sigma ** 2)
            + (2.0 / cfg.gamma) * np.log(1.0 + cfg.gamma / cfg.kappa)
        )

        # ── Compute bid/ask spreads from mid ─────────────────────────────
        # bid = mid + reservation_offset - spread/2
        # ask = mid + reservation_offset + spread/2
        # bid_spread_from_mid = -(reservation_offset - spread/2)
        #                     = -reservation_offset + spread/2
        # ask_spread_from_mid = reservation_offset + spread/2
        bid_spread = -reservation_offset + spread / 2.0
        ask_spread =  reservation_offset + spread / 2.0

        # Clamp to valid range.
        bid_spread = np.clip(bid_spread, cfg.min_spread, cfg.max_spread)
        ask_spread = np.clip(ask_spread, cfg.min_spread, cfg.max_spread)

        action = np.array([
            bid_spread,
            ask_spread,
            cfg.order_size,
            cfg.order_size,
        ], dtype=np.float64)

        # Dummy log_prob / value for interface compatibility.
        dummy_lp = torch.tensor(0.0)
        dummy_v = torch.tensor(0.0)

        return action, dummy_lp, dummy_v

    def get_spreads(self, inventory: float) -> tuple[float, float]:
        """
        Convenience: compute raw bid/ask spreads for a given inventory.

        Useful for testing the inventory-skew logic directly.

        Returns:
            (bid_spread, ask_spread) in tick-units (unclamped).
        """
        cfg = self.cfg
        reservation_offset = -inventory * cfg.gamma * (cfg.sigma ** 2)
        spread = (
            cfg.gamma * (cfg.sigma ** 2)
            + (2.0 / cfg.gamma) * np.log(1.0 + cfg.gamma / cfg.kappa)
        )
        bid_spread = -reservation_offset + spread / 2.0
        ask_spread =  reservation_offset + spread / 2.0
        return float(bid_spread), float(ask_spread)
