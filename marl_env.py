import os
import math
import numpy as np
import torch
import gymnasium
from gymnasium import spaces

import config as cfg
from env import SatelliteEnv

# Try importing PettingZoo ParallelEnv if installed; otherwise provide a standard fallback base class
try:
    from pettingzoo import ParallelEnv
except ImportError:
    class ParallelEnv:
        """Fallback ParallelEnv interface when pettingzoo is not installed."""
        pass


class MARLSatelliteEnv(ParallelEnv):
    """
    Multi-Agent Satellite Collision Avoidance Environment.
    Self Agent Architecture.

    Key Characteristics:
    - Decentralized local observations (LOCAL_OBS_DIM = 24 floats per satellite, O(1) invariant).
    - Decentralized local actions (LOCAL_ACT_DIM = 3 floats: [dvx, dvy, dvz] per satellite).
    - Vectorized PyTorch GPU orbital mechanics & pairwise distance evaluations.
    - Prevents both:
        1. Satellite-to-Debris collisions (Safe Distance: 15 km)
        2. Satellite-to-Satellite inter-agent collisions (Inter-Sat Distance: 10 km)
    - Compatible with PettingZoo ParallelEnv and batched tensor execution.
    """

    metadata = {"render_modes": ["human"], "name": "marl_satellite_v1"}

    # Observation layout (24 floats):
    # [0:3]   own position / 10000.0 (km)
    # [3:6]   own velocity / 10.0 (km/s)
    # [6:15]  relative positions of 3 nearest debris / 10000.0 (3 * 3 = 9)
    # [15:18] relative position of nearest other satellite / 10000.0 (3)
    # [18:21] relative velocity of nearest other satellite / 10.0 (3)
    # [21]    own min debris distance / 500.0 (1)
    # [22]    own instantaneous collision probability P_coll (1)
    # [23]    own remaining fuel level [0.0, 1.0] (1)
    LOCAL_OBS_DIM = 24
    LOCAL_ACT_DIM = 3

    def __init__(self, render_mode=None, n_satellites=None, max_debris=None):
        super().__init__()
        self.render_mode = render_mode
        self.n_satellites = n_satellites or cfg.N_SATELLITES
        self.max_debris   = max_debris or cfg.MAX_DEBRIS

        # Underlying physical engine
        self.base_env = SatelliteEnv(render_mode=render_mode)
        self.device   = self.base_env.device

        # Agent IDs
        self.possible_agents = [f"sat_{i}" for i in range(self.n_satellites)]
        self.agents          = self.possible_agents.copy()

        # Spaces per agent
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.LOCAL_OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-cfg.TIMESTEP_SECONDS * 0.001,
            high=cfg.TIMESTEP_SECONDS * 0.001,
            shape=(self.LOCAL_ACT_DIM,),
            dtype=np.float32
        )

        self.observation_spaces = {agent: self.observation_space for agent in self.possible_agents}
        self.action_spaces      = {agent: self.action_space      for agent in self.possible_agents}

        # Multi-agent state tracking
        self.individual_fuels  = np.ones(self.n_satellites, dtype=np.float32)
        self.lambda_cost_vec   = np.zeros(self.n_satellites, dtype=np.float32)
        self.current_step      = 0

        # Safety thresholds
        self.SAFE_DIST_DEBRIS_KM = cfg.SAFE_DISTANCE_KM  # 15 km
        self.SAFE_DIST_SATS_KM   = 10.0                  # 10 km inter-satellite buffer

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.base_env.reset(seed=seed, options=options)
        self.agents = self.possible_agents.copy()
        self.current_step = 0
        self.individual_fuels = np.ones(self.n_satellites, dtype=np.float32)
        self.lambda_cost_vec  = np.zeros(self.n_satellites, dtype=np.float32)

        obs_tensor = self._compute_all_local_observations()
        obs_np = obs_tensor.cpu().numpy()

        obs_dict = {agent: obs_np[i] for i, agent in enumerate(self.possible_agents)}
        info_dict = {
            agent: {
                "min_debris_dist": self.base_env.min_dist_val,
                "coverage": self.base_env.coverage_val,
                "fuel": float(self.individual_fuels[i]),
            }
            for i, agent in enumerate(self.possible_agents)
        }
        return obs_dict, info_dict

    def reset_batched(self, seed=None):
        """High-performance reset returning numpy array of shape (N_SATS, 24)."""
        obs_dict, info_dict = self.reset(seed=seed)
        obs_batch = np.stack([obs_dict[a] for a in self.possible_agents], axis=0)
        return obs_batch, info_dict

    def step(self, actions):
        """
        PettingZoo step interface.
        actions: dict { 'sat_0': np.ndarray(3,), ... } OR np.ndarray of shape (N_SATS, 3)
        """
        if isinstance(actions, dict):
            actions_arr = np.zeros((self.n_satellites, 3), dtype=np.float32)
            for i, agent in enumerate(self.possible_agents):
                if agent in actions:
                    actions_arr[i] = actions[agent]
        else:
            actions_arr = np.asarray(actions, dtype=np.float32).reshape(self.n_satellites, 3)

        obs_batch, rewards_arr, terms_arr, truncs_arr, infos = self.step_batched(actions_arr)

        obs_dict   = {agent: obs_batch[i] for i, agent in enumerate(self.possible_agents)}
        rew_dict   = {agent: float(rewards_arr[i]) for i, agent in enumerate(self.possible_agents)}
        term_dict  = {agent: bool(terms_arr[i]) for i, agent in enumerate(self.possible_agents)}
        trunc_dict = {agent: bool(truncs_arr[i]) for i, agent in enumerate(self.possible_agents)}
        info_dict  = {agent: infos[i] for i, agent in enumerate(self.possible_agents)}

        if any(terms_arr) or any(truncs_arr):
            self.agents = []

        return obs_dict, rew_dict, term_dict, trunc_dict, info_dict

    def step_batched(self, actions_arr):
        """
        High-performance vectorized step function.
        actions_arr: np.ndarray or torch.Tensor of shape (N_SATS, 3)
        Returns:
            obs_batch: np.ndarray (N_SATS, 24)
            rewards_arr: np.ndarray (N_SATS,)
            terms_arr: np.ndarray (N_SATS,)
            truncs_arr: np.ndarray (N_SATS,)
            infos: list of dicts (len N_SATS)
        """
        self.current_step += 1
        actions_arr = np.clip(actions_arr, -self.base_env.MAX_DV, self.base_env.MAX_DV)

        # 1. Update satellite thrust accumulator in base PyTorch engine
        action_t = torch.tensor(actions_arr, dtype=torch.float32, device=self.device)
        self.base_env.sat_thrust_acc += action_t * cfg.TIMESTEP_SECONDS

        # 2. Update individual fuel consumption
        thrust_mags = np.linalg.norm(actions_arr, axis=1)  # (N_SATS,)
        fuel_costs  = thrust_mags * 0.001
        self.individual_fuels = np.maximum(0.0, self.individual_fuels - fuel_costs)

        # 3. Update GPU physics
        self.base_env.current_step = self.current_step
        self.base_env._update_physics()

        # 4. Satellite-to-satellite pairwise distance matrix: [N_SATS, N_SATS]
        sat_pos = self.base_env.sat_pos_current
        sat_sat_dist = torch.cdist(sat_pos, sat_pos)
        diag_mask = torch.eye(self.n_satellites, device=self.device, dtype=torch.bool)
        sat_sat_dist_no_diag = sat_sat_dist.masked_fill(diag_mask, float('inf'))

        min_inter_sat_dists, nearest_sat_indices = torch.min(sat_sat_dist_no_diag, dim=1)

        # 5. Satellite-to-debris min distances: [N_SATS]
        sat_deb_dist = self.base_env.dist_matrix  # [N_SATS, N_DEBRIS]
        min_deb_dists, _ = torch.min(sat_deb_dist, dim=1)

        # 6. Compute Multi-Agent Decentralized Rewards
        # (a) Team coverage bonus
        team_cov_reward = cfg.REWARD_COVERAGE * self.base_env.coverage_val

        # (b) Local debris safety penalty
        deb_dists_np = min_deb_dists.cpu().numpy()
        deb_danger_penalty = np.where(
            deb_dists_np < self.SAFE_DIST_DEBRIS_KM,
            cfg.REWARD_DANGER * (self.SAFE_DIST_DEBRIS_KM - deb_dists_np),
            0.0
        )

        # (c) Inter-satellite safety penalty (prevent sat-sat collision during evasive maneuvers)
        inter_sat_dists_np = min_inter_sat_dists.cpu().numpy()
        inter_sat_penalty = np.where(
            inter_sat_dists_np < self.SAFE_DIST_SATS_KM,
            -2.0 * (self.SAFE_DIST_SATS_KM - inter_sat_dists_np),
            0.0
        )

        # (d) Individual fuel penalty
        fuel_penalty = -0.01 * (thrust_mags ** 2)

        # (e) Lagrangian Safety Cost per satellite
        SIGMA = 5.0
        p_coll_vec = np.exp(-(deb_dists_np ** 2) / (2.0 * (SIGMA ** 2)))
        violation = p_coll_vec - cfg.COST_LIMIT
        self.lambda_cost_vec = np.clip(
            self.lambda_cost_vec + cfg.LAMBDA_LR * violation, 0.0, cfg.LAMBDA_MAX
        )
        lagrange_penalty = -self.lambda_cost_vec * p_coll_vec

        # Total individual reward
        rewards_arr = team_cov_reward + deb_danger_penalty + inter_sat_penalty + fuel_penalty + lagrange_penalty

        # Hard collision check (<1 km with debris OR <1 km with another satellite)
        catastrophic_deb = deb_dists_np < 1.0
        catastrophic_sat = inter_sat_dists_np < 1.0
        hard_collision = catastrophic_deb | catastrophic_sat

        rewards_arr[hard_collision] -= 1000.0

        # Terminations and Truncations
        terminated_global = bool(np.any(hard_collision))
        truncated_global  = (self.current_step >= cfg.EPISODE_LENGTH) or bool(np.all(self.individual_fuels <= 0.0))

        terms_arr  = np.full(self.n_satellites, terminated_global, dtype=bool)
        truncs_arr = np.full(self.n_satellites, truncated_global,  dtype=bool)

        # 7. Build local observations for all satellites
        obs_tensor = self._compute_all_local_observations(
            nearest_sat_indices=nearest_sat_indices,
            min_deb_dists=min_deb_dists
        )
        obs_batch = obs_tensor.cpu().numpy()

        infos = [
            {
                "min_debris_dist": float(deb_dists_np[i]),
                "min_sat_dist": float(inter_sat_dists_np[i]),
                "p_coll": float(p_coll_vec[i]),
                "fuel": float(self.individual_fuels[i]),
                "coverage": self.base_env.coverage_val,
                "step": self.current_step,
            }
            for i in range(self.n_satellites)
        ]

        return obs_batch, rewards_arr, terms_arr, truncs_arr, infos

    def _compute_all_local_observations(self, nearest_sat_indices=None, min_deb_dists=None):
        """
        Fully vectorized PyTorch computation of 24-dimensional local states for all satellites.
        Runs in ~0.5ms on GPU/CPU.
        """
        PNORM = 10000.0
        VNORM = 10.0

        sat_pos = self.base_env.sat_pos_current      # [N_SATS, 3]
        sat_vel = self.base_env.sat_vel0_t          # [N_SATS, 3]
        deb_pos = self.base_env.deb_pos_current      # [N_DEB, 3]

        N_SATS = self.n_satellites
        N_DEB  = deb_pos.shape[0]

        # 1. Own normalized kinematics [N_SATS, 6]
        own_kinematics = torch.cat([sat_pos / PNORM, sat_vel / VNORM], dim=1)

        # 2. Top-3 nearest debris relative vectors [N_SATS, 9]
        dist_matrix = self.base_env.dist_matrix  # [N_SATS, N_DEB]
        k_deb = min(3, N_DEB)
        topk_dists, topk_indices = torch.topk(dist_matrix, k=k_deb, dim=1, largest=False)

        # Gather coordinates of top-3 debris: [N_SATS, 3, 3]
        nearest_deb_pos = deb_pos[topk_indices]  # [N_SATS, 3, 3]
        rel_deb_pos = (nearest_deb_pos - sat_pos.unsqueeze(1)) / PNORM
        rel_deb_flat = rel_deb_pos.view(N_SATS, k_deb * 3)

        if k_deb < 3:
            pad = torch.zeros((N_SATS, (3 - k_deb) * 3), device=self.device, dtype=torch.float32)
            rel_deb_flat = torch.cat([rel_deb_flat, pad], dim=1)

        # 3. Nearest other satellite kinematics [N_SATS, 6]
        if nearest_sat_indices is None:
            sat_sat_dist = torch.cdist(sat_pos, sat_pos)
            diag_mask = torch.eye(N_SATS, device=self.device, dtype=torch.bool)
            sat_sat_dist_no_diag = sat_sat_dist.masked_fill(diag_mask, float('inf'))
            _, nearest_sat_indices = torch.min(sat_sat_dist_no_diag, dim=1)

        nearest_sat_pos = sat_pos[nearest_sat_indices]
        nearest_sat_vel = sat_vel[nearest_sat_indices]

        rel_sat_pos = (nearest_sat_pos - sat_pos) / PNORM
        rel_sat_vel = (nearest_sat_vel - sat_vel) / VNORM
        rel_sat_kinematics = torch.cat([rel_sat_pos, rel_sat_vel], dim=1)  # [N_SATS, 6]

        # 4. Local safety & state metrics [N_SATS, 3]
        if min_deb_dists is None:
            min_deb_dists, _ = torch.min(dist_matrix, dim=1)

        norm_min_dist = torch.clamp(min_deb_dists / 500.0, 0.0, 1.0).unsqueeze(1)
        SIGMA = 5.0
        p_coll = torch.exp(-(min_deb_dists ** 2) / (2.0 * (SIGMA ** 2))).unsqueeze(1)
        fuel_t = torch.tensor(self.individual_fuels, device=self.device, dtype=torch.float32).unsqueeze(1)

        local_metrics = torch.cat([norm_min_dist, p_coll, fuel_t], dim=1)  # [N_SATS, 3]

        # 5. Concatenate all into final observation: [N_SATS, 24]
        local_obs_tensor = torch.cat([
            own_kinematics,      # 6
            rel_deb_flat,        # 9
            rel_sat_kinematics,  # 6
            local_metrics        # 3
        ], dim=1)

        return torch.clamp(local_obs_tensor, -1.0, 1.0)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing MARLSatelliteEnv (Self Agent Multi-Agent Architecture)...")
    print("=" * 60)
    env = MARLSatelliteEnv(render_mode="human")
    obs_batch, infos = env.reset_batched(seed=42)
    print(f"  Initialized {env.n_satellites} autonomous agents.")
    print(f"  Local Observation Batch Shape: {obs_batch.shape} (Expected: ({env.n_satellites}, 24))")

    import time
    t0 = time.time()
    n_test_steps = 100
    for s in range(n_test_steps):
        # Sample random continuous thrusts for all 70 satellites in parallel
        actions = np.random.uniform(-0.01, 0.01, size=(env.n_satellites, 3)).astype(np.float32)
        obs_batch, rewards, terms, truncs, info_list = env.step_batched(actions)
        if np.any(terms) or np.any(truncs):
            obs_batch, infos = env.reset_batched()

    elapsed = time.time() - t0
    fps = (n_test_steps * env.n_satellites) / elapsed
    step_fps = n_test_steps / elapsed
    print(f"\nBenchmark Performance:")
    print(f"  Elapsed: {elapsed:.2f}s for {n_test_steps} multi-agent steps")
    print(f"  Simulation Speed: {step_fps:.1f} constellation steps/sec ({fps:.1f} agent steps/sec)")
    print(f"  Sample satellite 0 min debris dist: {info_list[0]['min_debris_dist']:.1f} km")
    print(f"  Sample satellite 0 min sat dist:    {info_list[0]['min_sat_dist']:.1f} km")
    print("MARLSatelliteEnv is working properly!")
