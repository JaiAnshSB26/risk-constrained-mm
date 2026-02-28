# ============================================================================
# risk-constrained-mm :: python/rcmm/cppo.py
# ============================================================================
"""
Constrained Proximal Policy Optimization (CPPO) with CVaR tail-risk penalty.

Extends the baseline PPO (Phase 7) with a Lagrangian CVaR constraint so the
agent optimises expected return while bounding downside tail risk — critical
for surviving flash-crash regimes in market making.

Mathematical formulation:

    max_θ  E[R(τ)]                    (maximise expected return)
    s.t.   CVaR_α(R) ≥ d             (bound tail risk)

where:
    VaR_α   = inf{ z : P(Z ≤ z) ≥ α }           (α-quantile of returns)
    CVaR_α  = E[Z | Z ≤ VaR_α]                   (expected shortfall)

We relax the constraint via a Lagrangian multiplier λ:

    L_CPPO = L_PPO + λ · max(0, d - CVaR_α)

λ is updated by dual gradient ascent:
    λ ← max(0, λ + η_λ · (d - CVaR_α))

When the tail performs worse than threshold d, λ increases and the agent
gets penalised more heavily; when the tail is acceptable, λ shrinks to 0.

This implementation inherits the CleanRL-style single-file design from
ppo.py for hackability and extends it minimally for the constraint.

References:
    - Tamar et al., "Policy Gradient for Coherent Risk Measures" (NeurIPS 2015)
    - Chow et al., "Risk-Constrained RL with CVaR" (NeurIPS 2017)
    - Schulman et al., "Proximal Policy Optimization" (2017)
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from rcmm.ppo import ActorCritic, PPOConfig, RolloutBuffer


# ── CVaR Computation ─────────────────────────────────────────────────────────


def compute_cvar(
    rewards: torch.Tensor,
    alpha: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute empirical VaR and CVaR at confidence level α.

    Given a 1-D tensor of rewards (or returns) Z:
        VaR_α  = α-quantile (the cutoff for the worst α fraction)
        CVaR_α = mean(Z[Z ≤ VaR_α])  (expected shortfall / tail mean)

    If the tail is empty (degenerate case), CVaR falls back to VaR.

    Args:
        rewards: 1-D tensor of scalar rewards/returns.
        alpha: Tail probability (default 0.05 = bottom 5%).

    Returns:
        (var_alpha, cvar_alpha): both scalar tensors (differentiable
        through the mean of the tail, though VaR itself is not smooth).
    """
    n = rewards.numel()
    k = max(1, int(n * alpha))  # number of samples in the tail

    sorted_rewards, _ = torch.sort(rewards)
    var_alpha = sorted_rewards[k - 1]  # α-quantile

    # Tail: all rewards ≤ VaR (at least k samples).
    tail_mask = rewards <= var_alpha
    if tail_mask.sum() > 0:
        cvar_alpha = rewards[tail_mask].mean()
    else:
        cvar_alpha = var_alpha  # degenerate fallback

    return var_alpha, cvar_alpha


# ── CPPO Configuration ──────────────────────────────────────────────────────


@dataclasses.dataclass
class CPPOConfig(PPOConfig):
    """CPPO hyperparameters — extends PPOConfig with risk constraint params."""

    # CVaR parameters.
    cvar_alpha: float = 0.05        # tail probability (bottom 5%)
    cvar_threshold: float = -10.0   # d: acceptable CVaR floor
    lagrange_lr: float = 0.01       # η_λ: Lagrange multiplier learning rate
    lagrange_init: float = 0.1      # initial λ value
    lagrange_max: float = 10.0      # upper bound on λ to prevent instability
    cvar_penalty_coef: float = 1.0  # fixed weight (used if dual_update=False)
    dual_update: bool = True        # whether to use adaptive Lagrangian


# ── CPPO Agent ───────────────────────────────────────────────────────────────


class CPPOAgent:
    """
    Constrained PPO agent with CVaR tail-risk penalty.

    Inherits the PPO training loop and extends it with:
      1. CVaR computation over rollout rewards (bottom α-quantile).
      2. Lagrangian penalty: λ · max(0, d - CVaR_α) added to PPO loss.
      3. Dual gradient ascent on λ after each update.

    The network architecture is identical to baseline PPO (ActorCritic MLP).
    """

    def __init__(self, env: gym.Env, cfg: CPPOConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else CPPOConfig()
        self.device = torch.device(self.cfg.device)

        # Extract dimensions from env.
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert isinstance(env.action_space, gym.spaces.Box)
        self.obs_dim = int(np.prod(env.observation_space.shape))
        self.act_dim = int(np.prod(env.action_space.shape))

        # Action bounds.
        self.action_low = torch.as_tensor(
            env.action_space.low, dtype=torch.float32, device=self.device
        )
        self.action_high = torch.as_tensor(
            env.action_space.high, dtype=torch.float32, device=self.device
        )

        # Network (same architecture as baseline PPO).
        self.network = ActorCritic(
            self.obs_dim, self.act_dim, self.cfg
        ).to(self.device)

        # Optimizer.
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.cfg.learning_rate, eps=1e-5
        )

        # Rollout buffer.
        self.buffer = RolloutBuffer(
            self.cfg.rollout_steps, self.obs_dim, self.act_dim, self.cfg.device
        )

        # Lagrangian multiplier (non-negative).
        self._lagrange_multiplier = self.cfg.lagrange_init

    @property
    def lagrange_multiplier(self) -> float:
        """Current value of the Lagrangian multiplier λ."""
        return self._lagrange_multiplier

    def select_action(
        self, obs: np.ndarray
    ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """Select action — identical to baseline PPO."""
        obs_t = torch.as_tensor(
            obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            action, log_prob, _entropy, value = (
                self.network.get_action_and_value(obs_t)
            )

        action_np = action.squeeze(0).cpu().numpy()
        clipped = np.clip(
            action_np,
            self.action_low.cpu().numpy(),
            self.action_high.cpu().numpy(),
        )
        return clipped, log_prob.squeeze(0), value.squeeze(0)

    def update(self) -> dict[str, float]:
        """
        Run CPPO update: PPO objective + Lagrangian CVaR penalty.

        Returns dict of training metrics including CVaR diagnostics.
        """
        cfg = self.cfg
        buffer = self.buffer
        b_size = cfg.rollout_steps
        mb_size = cfg.minibatch_size

        # ── Compute CVaR over rollout rewards ──
        rollout_rewards = buffer.rewards[:b_size]
        var_alpha, cvar_alpha = compute_cvar(
            rollout_rewards, alpha=cfg.cvar_alpha
        )
        cvar_val = cvar_alpha.item()
        var_val = var_alpha.item()

        # CVaR constraint violation: how much worse than threshold.
        constraint_violation = cfg.cvar_threshold - cvar_val  # positive = bad

        # Lagrangian penalty weight.
        if cfg.dual_update:
            lam = self._lagrange_multiplier
        else:
            lam = cfg.cvar_penalty_coef

        # Flatten buffer data.
        b_obs = buffer.obs[:b_size]
        b_actions = buffer.actions[:b_size]
        b_log_probs = buffer.log_probs[:b_size]
        b_advantages = buffer.advantages[:b_size]
        b_returns = buffer.returns[:b_size]
        b_values = buffer.values[:b_size]

        # Normalise advantages.
        adv_mean = b_advantages.mean()
        adv_std = b_advantages.std()
        b_advantages = (b_advantages - adv_mean) / (adv_std + 1e-8)

        # Track metrics.
        total_pg_loss = 0.0
        total_v_loss = 0.0
        total_entropy = 0.0
        total_clipfrac = 0.0
        total_approx_kl = 0.0
        total_cvar_penalty = 0.0
        num_updates = 0

        for _epoch in range(cfg.update_epochs):
            indices = torch.randperm(b_size, device=self.device)

            for start in range(0, b_size, mb_size):
                end = start + mb_size
                mb_idx = indices[start:end]

                mb_obs = b_obs[mb_idx]
                mb_act = b_actions[mb_idx]
                mb_old_logp = b_log_probs[mb_idx]
                mb_adv = b_advantages[mb_idx]
                mb_ret = b_returns[mb_idx]
                mb_old_val = b_values[mb_idx]

                # Forward pass.
                _, new_logp, entropy, new_val = (
                    self.network.get_action_and_value(mb_obs, mb_act)
                )

                # ── Policy loss (clipped surrogate) ──
                log_ratio = new_logp - mb_old_logp
                ratio = torch.exp(log_ratio)

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()

                surr1 = ratio * mb_adv
                surr2 = torch.clamp(
                    ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps
                ) * mb_adv
                pg_loss = -torch.min(surr1, surr2).mean()

                # ── Value loss ──
                if cfg.clip_vloss:
                    v_clipped = mb_old_val + torch.clamp(
                        new_val - mb_old_val, -cfg.clip_eps, cfg.clip_eps
                    )
                    v_loss_unclipped = (new_val - mb_ret) ** 2
                    v_loss_clipped = (v_clipped - mb_ret) ** 2
                    v_loss = 0.5 * torch.max(
                        v_loss_unclipped, v_loss_clipped
                    ).mean()
                else:
                    v_loss = 0.5 * ((new_val - mb_ret) ** 2).mean()

                # ── Entropy loss ──
                entropy_loss = entropy.mean()

                # ── CVaR penalty ──
                # Penalise the minibatch proportional to how much the
                # rollout's CVaR violates the threshold.
                cvar_penalty = lam * max(0.0, constraint_violation)

                # ── Total CPPO loss ──
                loss = (
                    pg_loss
                    - cfg.entropy_coef * entropy_loss
                    + cfg.vf_coef * v_loss
                    + cvar_penalty
                )

                # ── Backward + step ──
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), cfg.max_grad_norm
                )
                self.optimizer.step()

                # Accumulate metrics.
                total_pg_loss += pg_loss.item()
                total_v_loss += v_loss.item()
                total_entropy += entropy_loss.item()
                total_approx_kl += approx_kl.item()
                total_cvar_penalty += cvar_penalty
                with torch.no_grad():
                    total_clipfrac += (
                        (torch.abs(ratio - 1.0) > cfg.clip_eps)
                        .float()
                        .mean()
                        .item()
                    )
                num_updates += 1

        # ── Dual variable update (Lagrangian multiplier) ──
        if cfg.dual_update:
            self._lagrange_multiplier = float(np.clip(
                self._lagrange_multiplier + cfg.lagrange_lr * constraint_violation,
                0.0,
                cfg.lagrange_max,
            ))

        # Average metrics.
        n = max(num_updates, 1)
        return {
            "policy_loss": total_pg_loss / n,
            "value_loss": total_v_loss / n,
            "entropy": total_entropy / n,
            "approx_kl": total_approx_kl / n,
            "clip_fraction": total_clipfrac / n,
            "cvar": cvar_val,
            "var": var_val,
            "cvar_penalty": total_cvar_penalty / n,
            "lagrange_lambda": self._lagrange_multiplier,
            "constraint_violation": constraint_violation,
        }

    def collect_rollout(
        self, env: gym.Env, obs: np.ndarray
    ) -> tuple[np.ndarray, bool]:
        """Collect a full rollout — identical to baseline PPO."""
        self.buffer.reset()
        done = False

        for _ in range(self.cfg.rollout_steps):
            action, log_prob, value = self.select_action(obs)

            next_obs, reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated

            self.buffer.add(
                obs, torch.as_tensor(action), reward, done, log_prob, value
            )

            if done:
                next_obs, _info = env.reset()
                done = False

            obs = next_obs

        # Compute GAE with bootstrap value.
        with torch.no_grad():
            last_obs_t = torch.as_tensor(
                obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            last_value = self.network.get_value(last_obs_t).squeeze()

        self.buffer.compute_gae(
            last_value, done, self.cfg.gamma, self.cfg.gae_lambda
        )

        return obs, done

    def train(
        self,
        env: gym.Env,
        total_timesteps: int,
        *,
        log_interval: int = 1,
        verbose: bool = True,
    ) -> list[dict[str, float]]:
        """
        Train the CPPO agent.

        Returns list of metric dicts (one per update), including CVaR
        diagnostics and Lagrange multiplier trajectory.
        """
        obs, _info = env.reset()
        num_updates = total_timesteps // self.cfg.rollout_steps
        all_metrics: list[dict[str, float]] = []

        for update in range(1, num_updates + 1):
            obs, _done = self.collect_rollout(env, obs)
            metrics = self.update()
            metrics["update"] = float(update)
            metrics["timesteps"] = float(update * self.cfg.rollout_steps)
            all_metrics.append(metrics)

            if verbose and update % log_interval == 0:
                print(
                    f"Update {update}/{num_updates} | "
                    f"steps={int(metrics['timesteps'])} | "
                    f"pg={metrics['policy_loss']:.4f} | "
                    f"vf={metrics['value_loss']:.4f} | "
                    f"CVaR={metrics['cvar']:.2f} | "
                    f"λ={metrics['lagrange_lambda']:.4f} | "
                    f"kl={metrics['approx_kl']:.4f}"
                )

        return all_metrics
