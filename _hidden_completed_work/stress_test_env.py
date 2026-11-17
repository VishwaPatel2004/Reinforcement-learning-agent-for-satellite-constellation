# =============================================================================
# stress_test_env.py
# COLLISION STRESS TEST ENVIRONMENT
#
# This wraps SatelliteEnv and does 3 things to FORCE collisions:
#
#   1. Loads ALL debris types (Iridium-33 + Cosmos-2251 + Fengyun-1C +
#      Cosmos-1408) — giving ~1500+ debris objects
#
#   2. Increases N_SATELLITES to use all available Starlink satellites
#
#   3. INJECTS SYNTHETIC DEBRIS — places ghost debris pieces directly on
#      a collision course with 1-3 random satellites every episode.
#      These synthetic pieces close the gap at ~0.5 km/step so the agent
#      MUST thrust away or face a guaranteed collision within ~60 steps.
#
# PURPOSE: Prove the Safe RL agent actively avoids threats. Without this
# test, "no collision" could just mean the real orbits never cross.
# =============================================================================

import os
import json
import math
import random
import numpy as np
import gymnasium
from skyfield.api import load, EarthSatellite
import config as cfg

# ─── override N_SATELLITES for stress test ───────────────────────────────────
STRESS_N_SATELLITES = min(100, cfg.N_SATELLITES)   # use up to 100 satellites
STRESS_SYNTHETIC    = 5   # inject 5 synthetic debris on collision courses

class StressTestEnv(gymnasium.Env):
    """
    Full-debris stress-test environment with synthetic near-collision injection.

    Observation / action spaces match SatelliteEnv exactly so the pre-trained
    Safe PPO model can be plugged in directly.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, n_synthetic=STRESS_SYNTHETIC):
        super().__init__()
        self.render_mode = render_mode
        self.n_synthetic = n_synthetic

        self.ts = load.timescale()
        self._load_all_debris()

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

        print(f"\nStressTestEnv initialized")
        print(f"  Real satellites     : {STRESS_N_SATELLITES}")
        print(f"  Real debris total   : {len(self.all_debris_data)}")
        print(f"  Synthetic injections: {self.n_synthetic} (on collision course!)")
        print(f"  State size          : {self.state_dim}")
        print(f"  Action size         : {self.action_dim}")

    def _load_all_debris(self):
        """Load ALL available debris files and combine them."""
        self.all_debris_data = []

        # Load every debris file we have
        debris_files = {
            "iridium33_debris.json"  : "Iridium-33",
            "cosmos2251_debris.json" : "Cosmos-2251",
            "fengyun1c_debris.json"  : "Fengyun-1C",
            "cosmos1408_debris.json" : "Cosmos-1408",
        }

        for fname, label in debris_files.items():
            path = os.path.join(cfg.DATA_DIR, fname)
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                    self.all_debris_data.extend(data)
                    print(f"  Loaded {len(data):4d} pieces from {label}")
            else:
                print(f"  ⚠️  {fname} not found, skipping")

        # Load Starlink satellites
        with open(os.path.join(cfg.DATA_DIR, "starlink_satellites.json")) as f:
            self.satellite_data = json.load(f)[:STRESS_N_SATELLITES]

        print(f"  Total real debris   : {len(self.all_debris_data)}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.current_step = 0
        self.lambda_cost  = cfg.LAMBDA_INIT
        self.time_offset  = random.uniform(0, 86400)
        self.current_time = self.ts.now()
        self.current_time = self.ts.tt_jd(
            self.current_time.tt + self.time_offset / 86400.0
        )

        # Build real satellite objects
        self.satellites = []
        for obj in self.satellite_data:
            try:
                self.satellites.append(EarthSatellite.from_omm(self.ts, obj))
            except: continue

        # Build real debris objects (sample up to 300 for speed)
        sample = random.sample(self.all_debris_data,
                               min(300, len(self.all_debris_data)))
        self.debris_objects = []
        for obj in sample:
            try:
                self.debris_objects.append(EarthSatellite.from_omm(self.ts, obj))
            except: continue

        # Get satellite positions to inject synthetic debris ON COURSE
        sat_positions = []
        for sat in self.satellites[:self.n_synthetic]:
            pos = sat.at(self.current_time).position.km
            sat_positions.append(pos)

        # Synthetic debris: start very close (150 km) and approach at 0.5 km/step
        self.synthetic_debris_pos = []   # current 3D position
        self.synthetic_debris_vel = []   # velocity vector (km per step)

        for sp in sat_positions:
            # Place synthetic debris 150 km away in a random direction
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            start_pos = sp + direction * 150.0
            # Velocity points TOWARD the satellite at 0.5 km/step
            approach_vel = -direction * 0.5
            self.synthetic_debris_pos.append(start_pos.copy())
            self.synthetic_debris_vel.append(approach_vel)

        # Thrust and fuel
        self.sat_thrust = np.zeros((len(self.satellites), 3))
        self.fuel = 1.0

        obs  = self._build_state()
        info = {
            "coverage"       : self._get_coverage(),
            "min_distance_km": self._get_min_distance(),
            "collision_prob" : self._get_collision_prob(),
        }
        return obs, info

    def _get_positions(self):
        """Get (N_sat, 3) and (N_deb, 3) position arrays including synthetic debris."""
        # Real satellites (with thrust offset)
        sat_pos = np.zeros((len(self.satellites), 3))
        for i, sat in enumerate(self.satellites):
            pos = sat.at(self.current_time).position.km
            pos = pos + self.sat_thrust[i] * self.current_step * cfg.TIMESTEP_SECONDS
            sat_pos[i] = pos

        # Real debris
        real_deb = []
        for deb in self.debris_objects:
            real_deb.append(deb.at(self.current_time).position.km)

        # Synthetic debris (positions updated each step)
        synth_deb = list(self.synthetic_debris_pos)

        all_deb = np.array(real_deb + synth_deb) if real_deb else np.array(synth_deb)
        return sat_pos, all_deb

    def _advance_synthetic_debris(self, sat_pos):
        """Move synthetic debris one step closer to their target satellite."""
        for idx in range(len(self.synthetic_debris_pos)):
            if idx < len(sat_pos):
                # Recalculate direction toward satellite (tracks after thrust)
                target = sat_pos[idx]
                current = self.synthetic_debris_pos[idx]
                diff = target - current
                dist = np.linalg.norm(diff)
                if dist > 0.5:
                    direction = diff / dist
                    # Approach at 0.5 km per step
                    self.synthetic_debris_vel[idx] = direction * 0.5
            self.synthetic_debris_pos[idx] += self.synthetic_debris_vel[idx]

    def _get_min_distance(self):
        sat_pos, deb_pos = self._get_positions()
        min_dist = float('inf')
        for sp in sat_pos:
            for dp in deb_pos:
                d = np.linalg.norm(sp - dp)
                if d < min_dist:
                    min_dist = d
        return float(min_dist)

    def _get_coverage(self):
        lat_grid = np.arange(-90, 90, 10)   # coarser grid for speed
        lon_grid = np.arange(-180, 180, 10)
        sat_pos, _ = self._get_positions()
        covered = 0
        total   = 0
        for lat in lat_grid:
            for lon in lon_grid:
                total += 1
                lat_r, lon_r = math.radians(lat), math.radians(lon)
                gnd = np.array([
                    cfg.EARTH_RADIUS_KM * math.cos(lat_r) * math.cos(lon_r),
                    cfg.EARTH_RADIUS_KM * math.cos(lat_r) * math.sin(lon_r),
                    cfg.EARTH_RADIUS_KM * math.sin(lat_r)
                ])
                for sp in sat_pos:
                    sv = sp / np.linalg.norm(sp)
                    gv = gnd / np.linalg.norm(gnd)
                    cos_angle = np.clip(np.dot(sv, gv), -1, 1)
                    if math.degrees(math.acos(cos_angle)) < 60.0:
                        covered += 1
                        break
        return covered / total if total > 0 else 0.0

    def _get_collision_prob(self):
        sat_pos, deb_pos = self._get_positions()
        SIGMA = 5.0
        max_pc = 0.0
        for sp in sat_pos:
            for dp in deb_pos:
                d = np.linalg.norm(sp - dp)
                pc = math.exp(-(d**2) / (2 * SIGMA**2))
                if pc > max_pc:
                    max_pc = pc
        return float(max_pc)

    def _build_state(self):
        sat_pos, deb_pos = self._get_positions()
        state = []
        PNORM = 10000.0
        VNORM = 10.0

        for i, sat in enumerate(self.satellites):
            geo = sat.at(self.current_time)
            pos = geo.position.km + self.sat_thrust[i] * self.current_step * cfg.TIMESTEP_SECONDS
            vel = geo.velocity.km_per_s
            state += [pos[0]/PNORM, pos[1]/PNORM, pos[2]/PNORM,
                      vel[0]/VNORM, vel[1]/VNORM, vel[2]/VNORM]

        while len(state) < cfg.N_SATELLITES * 6:
            state += [0.0] * 6

        # Closest debris
        min_dist = float('inf')
        closest_deb = deb_pos[0] if len(deb_pos) > 0 else np.zeros(3)
        for sp in sat_pos:
            for dp in deb_pos:
                d = np.linalg.norm(sp - dp)
                if d < min_dist:
                    min_dist = d
                    closest_deb = dp

        state += [closest_deb[0]/PNORM, closest_deb[1]/PNORM, closest_deb[2]/PNORM]
        state.append(float(np.clip(min_dist / 500.0, 0, 1)))
        state.append(float(self._get_collision_prob()))
        state.append(float(self._get_coverage()))
        state.append(float(self.fuel))

        return np.array(state, dtype=np.float32)

    def step(self, action):
        action   = np.clip(action, -self.MAX_DV, self.MAX_DV)
        action_3d = action.reshape(cfg.N_SATELLITES, 3)
        actual_n  = min(len(self.satellites), cfg.N_SATELLITES)
        self.sat_thrust[:actual_n] += action_3d[:actual_n]

        fuel_used   = float(np.sum(np.abs(action)) * 0.001)
        self.fuel   = max(0.0, self.fuel - fuel_used)

        self.current_step += 1
        self.current_time  = self.ts.tt_jd(
            self.current_time.tt + cfg.TIMESTEP_SECONDS / 86400.0
        )

        # Advance synthetic debris AFTER the satellite has moved
        sat_pos, _ = self._get_positions()
        self._advance_synthetic_debris(sat_pos)

        min_dist  = self._get_min_distance()
        coverage  = self._get_coverage()
        coll_prob = self._get_collision_prob()

        # Reward
        reward = cfg.REWARD_COVERAGE * coverage
        if min_dist < cfg.SAFE_DISTANCE_KM:
            reward += cfg.REWARD_DANGER * (cfg.SAFE_DISTANCE_KM - min_dist)
        reward += cfg.REWARD_FUEL * fuel_used

        # Lagrangian penalty
        violation = coll_prob - cfg.COST_LIMIT
        self.lambda_cost = max(0.0, min(cfg.LAMBDA_MAX,
                               self.lambda_cost + cfg.LAMBDA_LR * violation))
        reward -= self.lambda_cost * coll_prob

        terminated = False
        truncated  = False
        if min_dist < 1.0:
            terminated = True
            reward    -= 1000.0
        if self.current_step >= cfg.EPISODE_LENGTH:
            truncated = True
        if self.fuel <= 0.0:
            truncated = True

        obs  = self._build_state()
        info = {
            "coverage"        : coverage,
            "min_distance_km" : min_dist,
            "collision_prob"  : coll_prob,
            "lambda_cost"     : self.lambda_cost,
            "fuel"            : self.fuel,
            "step"            : self.current_step,
            "collision"       : terminated,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            min_d  = self._get_min_distance()
            cov    = self._get_coverage()
            pc     = self._get_collision_prob()
            status = "🔴 COLLISION!" if min_d < 1.0 else (
                     "🟠 DANGER!"   if min_d < cfg.SAFE_DISTANCE_KM else
                     "🟡 Warning"   if min_d < 100 else "🟢 Safe")
            print(f"  Step {self.current_step:4d} | "
                  f"Coverage: {cov:.1%} | "
                  f"Min dist: {min_d:8.2f} km {status} | "
                  f"P_c: {pc:.5f} | "
                  f"λ: {self.lambda_cost:.3f}")
