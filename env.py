import os
import json
import math
import random
import numpy as np
import gymnasium
import torch
from skyfield.api import load, EarthSatellite
import config as cfg

class SatelliteEnv(gymnasium.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"SatelliteEnv PyTorch Engine: {self.device}")

        self._load_orbital_data()
        self.ts = load.timescale()

        self.current_step = 0
        self.lambda_cost  = cfg.LAMBDA_INIT

        # Precompute the ground grid as a PyTorch tensor
        lat_grid = np.arange(-90, 90, 5) 
        lon_grid = np.arange(-180, 180, 5)
        self.n_grid = len(lat_grid) * len(lon_grid)
        
        gnd_grid = []
        for lat in lat_grid:
            for lon in lon_grid:
                lat_r = math.radians(lat)
                lon_r = math.radians(lon)
                gnd = [
                    cfg.EARTH_RADIUS_KM * math.cos(lat_r) * math.cos(lon_r),
                    cfg.EARTH_RADIUS_KM * math.cos(lat_r) * math.sin(lon_r),
                    cfg.EARTH_RADIUS_KM * math.sin(lat_r)
                ]
                gnd_grid.append(gnd)
                
        self.gnd_grid_tensor = torch.tensor(gnd_grid, dtype=torch.float32, device=self.device)
        self.gnd_norm_tensor = torch.nn.functional.normalize(self.gnd_grid_tensor, dim=1)
        self.cos_60 = math.cos(math.radians(60.0))

        self.state_dim  = cfg.N_SATELLITES * 6 + 3 + 4
        self.action_dim = cfg.N_SATELLITES * 3
        self.MAX_DV     = 0.01

        self.observation_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.state_dim,),
            dtype=np.float32
        )
        self.action_space = gymnasium.spaces.Box(
            low=-self.MAX_DV, high=self.MAX_DV,
            shape=(self.action_dim,),
            dtype=np.float32
        )
        print(f"SatelliteEnv initialized")
        print(f"Satellites  : {cfg.N_SATELLITES}")
        print(f"Debris      : {len(self.debris_data)}")
        print(f"State size  : {self.state_dim}")
        print(f"Action size : {self.action_dim}")
        print(f"Episode len : {cfg.EPISODE_LENGTH} steps")

    def _load_orbital_data(self):
        with open(os.path.join(cfg.DATA_DIR, "starlink_satellites.json"), 'r') as f:
            self.satellite_data = json.load(f)[:cfg.N_SATELLITES]
        
        self.debris_data = []
        debris_files = [
            "iridium33_debris.json",
            "cosmos2251_debris.json",
            "fengyun1c_debris.json",
            "cosmos1408_debris.json"
        ]
        for fname in debris_files:
            path = os.path.join(cfg.DATA_DIR, fname)
            if os.path.exists(path):
                with open(path, 'r') as f:
                    self.debris_data.extend(json.load(f))
        
        self.debris_data = self.debris_data[:cfg.MAX_DEBRIS]
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)

        self.current_step = 0
        self.time_offset = random.uniform(0, 86400)
        self.current_time = self.ts.now()
        self.current_time = self.ts.tt_jd(
            self.current_time.tt + self.time_offset / 86400.0
        )

        # 1. Evaluate Skyfield exactly ONCE at t=0
        sat_pos0 = []
        sat_vel0 = []
        for obj in self.satellite_data:
            try:
                sat = EarthSatellite.from_omm(self.ts, obj)
                geo = sat.at(self.current_time)
                pos = geo.position.km
                vel = geo.velocity.km_per_s
                if np.isnan(pos).any() or np.isnan(vel).any():
                    continue
                sat_pos0.append(pos)
                sat_vel0.append(vel)
            except:
                continue

        # Ensure we have N_SATELLITES exact
        while len(sat_pos0) < cfg.N_SATELLITES:
            sat_pos0.append([0.0, 0.0, 0.0])
            sat_vel0.append([0.0, 0.0, 0.0])
            
        sat_pos0 = sat_pos0[:cfg.N_SATELLITES]
        sat_vel0 = sat_vel0[:cfg.N_SATELLITES]

        deb_pos0 = []
        deb_vel0 = []
        for obj in self.debris_data:
            try:
                deb = EarthSatellite.from_omm(self.ts, obj)
                geo = deb.at(self.current_time)
                pos = geo.position.km
                vel = geo.velocity.km_per_s
                if np.isnan(pos).any() or np.isnan(vel).any():
                    continue
                deb_pos0.append(pos)
                deb_vel0.append(vel)
            except:
                continue

        if len(deb_pos0) == 0:
            # Fallback if no debris loaded
            deb_pos0.append([99999.0, 99999.0, 99999.0])
            deb_vel0.append([0.0, 0.0, 0.0])

        # 2. Transfer to GPU as PyTorch tensors
        self.sat_pos0_t = torch.tensor(np.array(sat_pos0), dtype=torch.float32, device=self.device)
        self.sat_vel0_t = torch.tensor(np.array(sat_vel0), dtype=torch.float32, device=self.device)
        self.deb_pos0_t = torch.tensor(np.array(deb_pos0), dtype=torch.float32, device=self.device)
        self.deb_vel0_t = torch.tensor(np.array(deb_vel0), dtype=torch.float32, device=self.device)

        self.sat_thrust_acc = torch.zeros((cfg.N_SATELLITES, 3), dtype=torch.float32, device=self.device)
        self.fuel = 1.0

        # Calculate initial step logic using tensor physics
        self._update_physics()

        obs = self._build_state()

        info = {
            "coverage": self.coverage_val,
            "min_distance_km": self.min_dist_val,
            "collision_prob": self.coll_prob_val,
        }
        return obs, info
        
    def _update_physics(self):
        """Propagates orbits linearly and computes metrics using PyTorch GPU ops."""
        t_sec = self.current_step * cfg.TIMESTEP_SECONDS
        
        # 1. Update positions
        self.sat_pos_current = self.sat_pos0_t + (self.sat_vel0_t * t_sec) + self.sat_thrust_acc
        self.deb_pos_current = self.deb_pos0_t + (self.deb_vel0_t * t_sec)

        # 2. Distance Matrix (Shape: [N_SATS, N_DEBRIS])
        self.dist_matrix = torch.cdist(self.sat_pos_current, self.deb_pos_current)
        self.min_dist_val = self.dist_matrix.min().item()

        # 3. Collision Probability (Max across all pairs)
        SIGMA = 5.0
        pc_matrix = torch.exp(-(self.dist_matrix**2) / (2 * SIGMA**2))
        self.coll_prob_val = pc_matrix.max().item()

        # 4. Coverage Calculation
        sat_norm = torch.nn.functional.normalize(self.sat_pos_current, dim=1)
        cos_angles = torch.matmul(sat_norm, self.gnd_norm_tensor.T) # [70, N_grid]
        covered_grid = (cos_angles > self.cos_60).any(dim=0)        # [N_grid]
        self.coverage_val = covered_grid.float().mean().item()

    def step(self, action):
        action = np.clip(action, -self.MAX_DV, self.MAX_DV)
        
        # Convert action to tensor and accumulate thrust
        action_t = torch.tensor(action, dtype=torch.float32, device=self.device).view(cfg.N_SATELLITES, 3)
        self.sat_thrust_acc += action_t * cfg.TIMESTEP_SECONDS

        fuel_used = float(np.sum(np.abs(action)) * 0.001)
        self.fuel  = max(0.0, self.fuel - fuel_used)

        self.current_step += 1
        
        # Update physics fully on the GPU
        self._update_physics()

        reward = cfg.REWARD_COVERAGE * self.coverage_val

        if self.min_dist_val < cfg.SAFE_DISTANCE_KM:
            reward += cfg.REWARD_DANGER * (cfg.SAFE_DISTANCE_KM - self.min_dist_val)

        reward += cfg.REWARD_FUEL * fuel_used

        violation = self.coll_prob_val - cfg.COST_LIMIT
        self.lambda_cost = max(0.0, min(cfg.LAMBDA_MAX,
                               self.lambda_cost + cfg.LAMBDA_LR * violation))
        reward -= self.lambda_cost * self.coll_prob_val

        terminated = False
        truncated  = False

        if self.min_dist_val < 1.0:
            terminated = True
            reward    -= 1000.0

        if self.current_step >= cfg.EPISODE_LENGTH:
            truncated = True

        if self.fuel <= 0.0:
            truncated = True

        obs = self._build_state()

        info = {
            "coverage":       self.coverage_val,
            "min_distance_km": self.min_dist_val,
            "collision_prob": self.coll_prob_val,
            "lambda_cost":    self.lambda_cost,
            "fuel":           self.fuel,
            "step":           self.current_step,
        }

        return obs, reward, terminated, truncated, info
    
    def _build_state(self):
        PNORM = 10000.0
        VNORM = 10.0

        # Flatten sat positions and velocities: [N_SATS, 6]
        sat_pos_norm = self.sat_pos_current / PNORM
        sat_vel_norm = self.sat_vel0_t / VNORM
        sat_state = torch.cat([sat_pos_norm, sat_vel_norm], dim=1).flatten()

        # Get the single globally closest debris coordinate
        min_idx_1d = torch.argmin(self.dist_matrix)
        closest_deb_idx = min_idx_1d % self.dist_matrix.shape[1]
        closest_deb = self.deb_pos_current[closest_deb_idx] / PNORM
        
        # Scalar metrics
        metrics = torch.tensor([
            np.clip(self.min_dist_val / 500.0, 0.0, 1.0),
            self.coll_prob_val,
            self.coverage_val,
            self.fuel
        ], dtype=torch.float32, device=self.device)

        # Concatenate on GPU then move to CPU
        state_t = torch.cat([sat_state, closest_deb, metrics])
        return state_t.cpu().numpy()

    def render(self):
        if self.render_mode == "human":
            status = " DANGER!" if self.min_dist_val < cfg.SAFE_DISTANCE_KM else (
                     " Warning" if self.min_dist_val < cfg.DANGER_ZONE_KM else " Safe")
            print(f"  Step {self.current_step:4d} | "
                  f"Coverage: {self.coverage_val:.1%} | "
                  f"Min dist: {self.min_dist_val:7.1f} km {status} | "
                  f"P_c: {self.coll_prob_val:.5f} | "
                  f"lambda: {self.lambda_cost:.3f} | "
                  f"Fuel: {self.fuel:.1%}")

if __name__ == "__main__":
    print("Testing env.py skeleton with PyTorch GPU Acceleration...")
    env = SatelliteEnv(render_mode="human")
    obs, info = env.reset()
    for _ in range(3):
        action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)
        env.render()
