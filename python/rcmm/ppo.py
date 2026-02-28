# ============================================================================
# risk-constrained-mm :: python/rcmm/ppo.py
# ============================================================================
"""
Clean, hackable, single-file PPO implementation (CleanRL-style).

Designed for continuous action spaces (market-making: spreads + sizes).
The Actor outputs mean + log_std for a diagonal Gaussian policy.
Designed to be straightforward to extend into Risk-Constrained PPO (Phase 8).

Architecture:
    Actor-Critic MLP with shared observation encoder:

    obs (22-d) ──► [Linear(obs_dim, 64) → Tanh
                    Linear(64, 64)       → Tanh] ──► shared features
                                                          │
                            ┌─────────────────────────────┤
                            ▼                             ▼
                     Actor head                    Critic head
                   Linear(64, act_dim)           Linear(64, 1)
                      = action mean                 = V(s)
                   + learnable log_std

    Policy: π(a|s) = N(mean(s), diag(exp(log_std)))
    Value:  V(s) = critic_head(features)

References:
    - Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
    - CleanRL: https://github.com/voidflight/cleanrl
    - Generalized Advantage Estimation: Schulman et al. (2015)
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


# ── Hyperparameters ──────────────────────────────────────────────────────────


@dataclasses.dataclass
class PPOConfig:
    """All PPO hyperparameters in one place."""

    # Network architecture.
    hidden_dim: int = 64
    num_hidden: int = 2

    # Training.
    learning_rate: float = 3e-4
    gamma: float = 0.99                 # discount factor
    gae_lambda: float = 0.95            # GAE λ
    clip_eps: float = 0.2               # PPO clipping ε
    clip_vloss: bool = True             # clip value loss
    entropy_coef: float = 0.01          # entropy bonus coefficient
    vf_coef: float = 0.5                # value loss coefficient
    max_grad_norm: float = 0.5          # gradient clipping

    # Rollout.
    rollout_steps: int = 2048           # steps per rollout
    num_minibatches: int = 32           # minibatches per update epoch
    update_epochs: int = 10             # PPO update epochs

    # Action scaling.
    action_low: Optional[np.ndarray] = None
    action_high: Optional[np.ndarray] = None

    # Misc.
    seed: int = 42
    device: str = "cpu"

    @property
    def minibatch_size(self) -> int:
        return self.rollout_steps // self.num_minibatches


# ── Actor-Critic Network ────────────────────────────────────────────────────


class ActorCritic(nn.Module):
    """
    Actor-Critic MLP for continuous action spaces.

    The actor outputs the mean of a diagonal Gaussian.  log_std is a
    learnable parameter (state-independent), following CleanRL convention.
    """

    def __init__(self, obs_dim: int, act_dim: int, cfg: PPOConfig) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # --- Shared feature extractor ---
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for _ in range(cfg.num_hidden):
            layers.append(nn.Linear(in_dim, cfg.hidden_dim))
            layers.append(nn.Tanh())
            in_dim = cfg.hidden_dim
        self.shared = nn.Sequential(*layers)

        # --- Actor head (mean) ---
        self.actor_mean = nn.Linear(cfg.hidden_dim, act_dim)

        # --- Learnable log_std (state-independent) ---
        self.actor_logstd = nn.Parameter(torch.zeros(act_dim))

        # --- Critic head ---
        self.critic = nn.Linear(cfg.hidden_dim, 1)

        # Orthogonal initialization (CleanRL best practice).
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.shared:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0.0)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0.0)

    def get_features(self, obs: torch.Tensor) -> torch.Tensor:
        """Run observation through the shared encoder."""
        return self.shared(obs)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """V(s) — state value estimate."""
        return self.critic(self.get_features(obs))

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample (or evaluate) action and return value.

        Returns:
            action:    sampled action (or the provided one)
            log_prob:  log π(a|s)
            entropy:   H[π(·|s)]
            value:     V(s)
        """
        features = self.get_features(obs)
        action_mean = self.actor_mean(features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)

        dist = Normal(action_mean, action_std)

        if action is None:
            action = dist.rsample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(features).squeeze(-1)

        return action, log_prob, entropy, value


# ── Rollout Buffer ───────────────────────────────────────────────────────────


class RolloutBuffer:
    """
    Fixed-size buffer for PPO rollout data.

    Stores transitions (obs, action, reward, done, log_prob, value)
    for one rollout of `rollout_steps` steps.
    """

    def __init__(self, rollout_steps: int, obs_dim: int, act_dim: int,
                 device: str = "cpu") -> None:
        self.rollout_steps = rollout_steps
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device
        self.pos = 0

        # Pre-allocate storage.
        self.obs = torch.zeros(rollout_steps, obs_dim, device=device)
        self.actions = torch.zeros(rollout_steps, act_dim, device=device)
        self.rewards = torch.zeros(rollout_steps, device=device)
        self.dones = torch.zeros(rollout_steps, device=device)
        self.log_probs = torch.zeros(rollout_steps, device=device)
        self.values = torch.zeros(rollout_steps, device=device)

        # Computed after rollout.
        self.advantages = torch.zeros(rollout_steps, device=device)
        self.returns = torch.zeros(rollout_steps, device=device)

    def add(
        self,
        obs: np.ndarray,
        action: torch.Tensor,
        reward: float,
        done: bool,
        log_prob: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Store one transition."""
        idx = self.pos
        self.obs[idx] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.actions[idx] = action.detach()
        self.rewards[idx] = reward
        self.dones[idx] = float(done)
        self.log_probs[idx] = log_prob.detach()
        self.values[idx] = value.detach()
        self.pos += 1

    def reset(self) -> None:
        """Reset position for next rollout."""
        self.pos = 0

    def is_full(self) -> bool:
        return self.pos >= self.rollout_steps

    def compute_gae(
        self,
        last_value: torch.Tensor,
        last_done: bool,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """
        Compute Generalized Advantage Estimation (GAE-λ).

        GAE(γ,λ):
            δ_t = r_t + γ·V(s_{t+1})·(1-d_{t+1}) - V(s_t)
            A_t = Σ_{l=0}^{T-t-1} (γλ)^l · δ_{t+l}
        """
        last_gae = torch.tensor(0.0, device=self.device)
        n = self.rollout_steps

        for t in reversed(range(n)):
            if t == n - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value.detach()
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

            delta = (
                self.rewards[t]
                + gamma * next_value * next_non_terminal
                - self.values[t]
            )
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values


# ── PPO Agent ────────────────────────────────────────────────────────────────


class PPOAgent:
    """
    Proximal Policy Optimization agent for continuous action spaces.

    Implements:
      - Clipped surrogate objective
      - Clipped value loss (optional)
      - Entropy bonus
      - Generalized Advantage Estimation (GAE)
      - Action rescaling to environment bounds

    Designed for easy extension to Constrained PPO (Phase 8).
    """

    def __init__(self, env: gym.Env, cfg: PPOConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else PPOConfig()
        self.device = torch.device(self.cfg.device)

        # Extract dimensions from env.
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert isinstance(env.action_space, gym.spaces.Box)
        self.obs_dim = int(np.prod(env.observation_space.shape))
        self.act_dim = int(np.prod(env.action_space.shape))

        # Action bounds for rescaling.
        self.action_low = torch.as_tensor(
            env.action_space.low, dtype=torch.float32, device=self.device
        )
        self.action_high = torch.as_tensor(
            env.action_space.high, dtype=torch.float32, device=self.device
        )

        # Network.
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

    def select_action(
        self, obs: np.ndarray
    ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """
        Select action given observation.

        Returns:
            clipped_action: numpy action clipped to env bounds
            log_prob:       log probability (for buffer)
            value:          V(s) estimate (for buffer)
        """
        obs_t = torch.as_tensor(
            obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            action, log_prob, _entropy, value = (
                self.network.get_action_and_value(obs_t)
            )

        # Clip to action space bounds.
        action_np = action.squeeze(0).cpu().numpy()
        clipped = np.clip(
            action_np,
            self.action_low.cpu().numpy(),
            self.action_high.cpu().numpy(),
        )

        return clipped, log_prob.squeeze(0), value.squeeze(0)

    def update(self) -> dict[str, float]:
        """
        Run PPO update using collected rollout data.

        Returns dict of training metrics.
        """
        cfg = self.cfg
        buffer = self.buffer
        b_size = cfg.rollout_steps
        mb_size = cfg.minibatch_size

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
        num_updates = 0

        for _epoch in range(cfg.update_epochs):
            # Shuffle indices.
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

                # Approximate KL for monitoring.
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

                # ── Total loss ──
                loss = (
                    pg_loss
                    - cfg.entropy_coef * entropy_loss
                    + cfg.vf_coef * v_loss
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
                with torch.no_grad():
                    total_clipfrac += (
                        (torch.abs(ratio - 1.0) > cfg.clip_eps)
                        .float()
                        .mean()
                        .item()
                    )
                num_updates += 1

        # Average metrics.
        n = max(num_updates, 1)
        return {
            "policy_loss": total_pg_loss / n,
            "value_loss": total_v_loss / n,
            "entropy": total_entropy / n,
            "approx_kl": total_approx_kl / n,
            "clip_fraction": total_clipfrac / n,
        }

    def collect_rollout(self, env: gym.Env, obs: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        Collect a full rollout of `rollout_steps` transitions.

        Args:
            env: Gymnasium environment.
            obs: Current observation.

        Returns:
            next_obs: observation after the rollout
            next_done: whether the episode is done
        """
        self.buffer.reset()
        done = False

        for _ in range(self.cfg.rollout_steps):
            action, log_prob, value = self.select_action(obs)

            next_obs, reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated

            self.buffer.add(obs, torch.as_tensor(action), reward, done, log_prob, value)

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

        self.buffer.compute_gae(last_value, done, self.cfg.gamma, self.cfg.gae_lambda)

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
        Train the PPO agent.

        Args:
            env: Gymnasium environment.
            total_timesteps: Total environment steps to train for.
            log_interval: Print metrics every N updates.
            verbose: Whether to print progress.

        Returns:
            List of metric dicts, one per update.
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
                    f"pg_loss={metrics['policy_loss']:.4f} | "
                    f"v_loss={metrics['value_loss']:.4f} | "
                    f"entropy={metrics['entropy']:.4f} | "
                    f"kl={metrics['approx_kl']:.4f}"
                )

        return all_metrics
