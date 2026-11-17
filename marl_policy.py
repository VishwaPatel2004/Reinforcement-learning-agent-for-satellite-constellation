import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

import config as cfg


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal initialization for stable PPO convergence."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class MAPPOActor(nn.Module):
    """
    Decentralized Actor Network.
    Takes satellite i's 24-dimensional local observation -> outputs 3D thrust vector.
    Architecture: 24 -> 128 (Tanh) -> 128 (Tanh) -> 3
    """
    def __init__(self, obs_dim=24, act_dim=3, max_action=0.01):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
        )
        self.mean_layer = layer_init(nn.Linear(128, act_dim), std=0.01)
        self.log_std    = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        features = self.net(obs)
        mean     = torch.tanh(self.mean_layer(features)) * self.max_action
        log_std  = torch.clamp(self.log_std, -20.0, 2.0)
        std      = log_std.exp().expand_as(mean)
        return Normal(mean, std)


class MAPPOCritic(nn.Module):
    """
    Critic Network (Value function V(s)).
    Takes local observation (or centralized state) -> outputs scalar expected return.
    """
    def __init__(self, obs_dim=24):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0)
        )

    def forward(self, obs):
        return self.net(obs)


class SafeMAPPO:
    """
    Multi-Agent Proximal Policy Optimization (MAPPO) with Lagrangian CMDP.
    Shared-parameter policy across all constellation satellites.
    Provides standard .predict() interface for easy drop-in use in evaluations and dashboards.
    """
    def __init__(
        self,
        obs_dim=24,
        act_dim=3,
        max_action=0.01,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device=None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_dim    = obs_dim
        self.act_dim    = act_dim
        self.max_action = max_action

        self.gamma         = gamma
        self.gae_lambda    = gae_lambda
        self.clip_coef     = clip_coef
        self.ent_coef      = ent_coef
        self.vf_coef       = vf_coef
        self.max_grad_norm = max_grad_norm

        self.actor  = MAPPOActor(obs_dim, act_dim, max_action).to(self.device)
        self.critic = MAPPOCritic(obs_dim).to(self.device)

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr,
            eps=1e-5
        )

        # Lagrangian Multiplier for Safety Constraint
        self.lambda_cost = cfg.LAMBDA_INIT
        self.cost_limit  = cfg.COST_LIMIT

    def get_action(self, obs_t, deterministic=False):
        """
        obs_t: tensor of shape (N_AGENTS, obs_dim)
        Returns: action_t, log_prob_t, value_t
        """
        dist  = self.actor(obs_t)
        value = self.critic(obs_t).squeeze(-1)

        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()

        action   = torch.clamp(action, -self.max_action, self.max_action)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value

    def evaluate_actions(self, obs_t, actions_t):
        """Evaluates batch of state-action pairs for PPO updates."""
        dist     = self.actor(obs_t)
        value    = self.critic(obs_t).squeeze(-1)
        log_prob = dist.log_prob(actions_t).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)
        return value, log_prob, entropy

    def update(self, b_obs, b_actions, b_logprobs, b_returns, b_advantages, b_costs=None, n_epochs=10, batch_size=256):
        """
        Executes standard PPO gradient descent over mini-batches of collected multi-agent experience.
        """
        b_obs        = torch.as_tensor(b_obs, dtype=torch.float32, device=self.device)
        b_actions    = torch.as_tensor(b_actions, dtype=torch.float32, device=self.device)
        b_logprobs   = torch.as_tensor(b_logprobs, dtype=torch.float32, device=self.device)
        b_returns    = torch.as_tensor(b_returns, dtype=torch.float32, device=self.device)
        b_advantages = torch.as_tensor(b_advantages, dtype=torch.float32, device=self.device)

        # Normalize advantages across batch
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        dataset_size = b_obs.shape[0]
        indices = np.arange(dataset_size)

        total_pg_loss = 0.0
        total_v_loss  = 0.0
        total_entropy = 0.0
        updates_count = 0

        for _ in range(n_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                mb_idx = indices[start:end]

                mb_obs        = b_obs[mb_idx]
                mb_actions    = b_actions[mb_idx]
                mb_old_logp   = b_logprobs[mb_idx]
                mb_returns    = b_returns[mb_idx]
                mb_adv        = b_advantages[mb_idx]

                new_value, new_logprob, entropy = self.evaluate_actions(mb_obs, mb_actions)

                # Ratio r_t
                logratio = new_logprob - mb_old_logp
                ratio    = logratio.exp()

                # Policy Loss with PPO clipping
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss
                v_loss = 0.5 * ((new_value - mb_returns) ** 2).mean()

                # Entropy Loss
                entropy_loss = entropy.mean()

                loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()

                total_pg_loss += pg_loss.item()
                total_v_loss  += v_loss.item()
                total_entropy += entropy_loss.item()
                updates_count += 1

        # Update Lagrangian multiplier based on episode costs if provided
        if b_costs is not None:
            mean_cost = float(np.mean(b_costs))
            violation = mean_cost - self.cost_limit
            self.lambda_cost = float(np.clip(
                self.lambda_cost + cfg.LAMBDA_LR * violation, 0.0, cfg.LAMBDA_MAX
            ))

        return {
            "pg_loss": total_pg_loss / max(1, updates_count),
            "v_loss":  total_v_loss / max(1, updates_count),
            "entropy": total_entropy / max(1, updates_count),
            "lambda":  self.lambda_cost,
        }

    def predict(self, obs, deterministic=True):
        """
        Drop-in inference method matching Stable-Baselines3 API.
        obs: numpy array of shape (N_SATS, 24) or (24,)
        Returns: action: numpy array of shape (N_SATS, 3) or (3,), state: None
        """
        is_single = (obs.ndim == 1)
        if is_single:
            obs = obs[np.newaxis, :]

        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            action_t, _, _ = self.get_action(obs_t, deterministic=deterministic)
            action = action_t.cpu().numpy()

        if is_single:
            action = action[0]

        return action, None

    def save(self, filepath):
        """Saves weights and optimizer state to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            "actor_state_dict":  self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lambda_cost": self.lambda_cost,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "max_action": self.max_action,
        }, filepath)

    @classmethod
    def load(cls, filepath, device=None):
        """Loads model weights from disk."""
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(filepath, map_location=device)

        model = cls(
            obs_dim=checkpoint.get("obs_dim", 24),
            act_dim=checkpoint.get("act_dim", 3),
            max_action=checkpoint.get("max_action", 0.01),
            device=device
        )
        model.actor.load_state_dict(checkpoint["actor_state_dict"])
        model.critic.load_state_dict(checkpoint["critic_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        model.lambda_cost = checkpoint.get("lambda_cost", 0.0)
        return model


if __name__ == "__main__":
    print("Testing SafeMAPPO module...")
    mappo = SafeMAPPO()
    sample_obs = np.random.uniform(-1, 1, size=(70, 24)).astype(np.float32)
    action, _ = mappo.predict(sample_obs)
    print("Sample action shape:", action.shape, "(Expected: (70, 3))")
    print(f"Action range: [{action.min():.5f}, {action.max():.5f}] km/s")
    print("SafeMAPPO model initialized successfully!")
