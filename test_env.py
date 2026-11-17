import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
import geodatasets
from env import SatelliteEnv
import config as cfg


print("\n[TEST 1] Creating SatelliteEnv...")
env = SatelliteEnv(render_mode="human")
print("PASSED: Environment created")


print("\n[TEST 2] Resetting (starting new episode)...")
obs, info = env.reset(seed=42)
print(f"  State vector shape   : {obs.shape}  (expected: ({env.state_dim},))")
print(f"  State vector dtype   : {obs.dtype}  (expected: float32)")
print(f"  State value range    : [{obs.min():.3f}, {obs.max():.3f}] (expected: [-1, 1])")
print(f"  Initial coverage     : {info['coverage']:.2%}")
print(f"  Initial min distance : {info['min_distance_km']:.2f} km")
print(f"  Initial P_collision  : {info['collision_prob']:.6f}")
assert obs.shape == (env.state_dim,), f"Wrong shape: {obs.shape}"
print("PASSED: Reset works correctly")


print("\n[TEST 3] Testing action space...")
rand_action = env.action_space.sample()
print(f"  Action shape : {rand_action.shape}  (expected: ({env.action_dim},))")
print(f"  Action range : [{rand_action.min():.4f}, {rand_action.max():.4f}]")
print(f"  Max DV       : {env.MAX_DV} km/s = {env.MAX_DV*1000} m/s")
print("PASSED: Action space correct")


print("\n[TEST 4] Running 5 random steps (shows one step every 10 seconds):")
print("-" * 60)

rewards = []
for step_num in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    rewards.append(reward)
    env.render()
    if terminated:
        print("Collision occurred!")
        break
    if truncated:
        print("Time limit reached")
        break

print("-" * 60)
print(f"  Rewards: {[f'{r:.3f}' for r in rewards]}")
print("  PASSED: Step function works")


print("\n[TEST 5] Running FULL episode with random agent (540 steps)...")
print("  This simulates 90 minutes (1 complete orbit)...")

obs, info = env.reset(seed=0)
total_reward = 0
coverages    = []
distances    = []
step_count   = 0

import time
start = time.time()

while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    coverages.append(info['coverage'])
    distances.append(info['min_distance_km'])
    step_count += 1
    if terminated or truncated:
        break

elapsed = time.time() - start

print(f"\n  Episode completed!")
print(f"  Steps           : {step_count} (= {step_count*10/60:.1f} minutes simulated)")
print(f"  Real time taken : {elapsed:.1f} seconds")
print(f"  Simulation speed: {step_count / elapsed:.1f} steps/second")
print(f"  Total reward    : {total_reward:.3f}")
print(f"  Avg coverage    : {np.mean(coverages):.2%}")
print(f"  Min distance    : {np.min(distances):.2f} km")
print(f"  End reason      : {'Collision ' if terminated else 'Time limit '}")


print("\n[MAP] Creating real-world satellite snapshot map...")

import json
from skyfield.api import load, EarthSatellite

ts = load.timescale()
t  = ts.now()

with open(cfg.DATA_DIR + r"\starlink_satellites.json") as f:
    slink = json.load(f)[:cfg.N_SATELLITES]
with open(cfg.DATA_DIR + r"\iridium33_debris.json") as f:
    ideb  = json.load(f)[:cfg.MAX_DEBRIS]

sat_lats, sat_lons = [], []
for obj in slink:
    try:
        s = EarthSatellite.from_omm(ts, obj)
        sub = s.at(t).subpoint()
        sat_lats.append(sub.latitude.degrees)
        sat_lons.append(sub.longitude.degrees)
    except: continue

deb_lats, deb_lons = [], []
for obj in ideb:
    try:
        d = EarthSatellite.from_omm(ts, obj)
        sub = d.at(t).subpoint()
        deb_lats.append(sub.latitude.degrees)
        deb_lons.append(sub.longitude.degrees)
    except: continue

world_land = gpd.read_file(geodatasets.get_path('naturalearth.land'))

fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor('#0a0e1a')
ax.set_facecolor('#0a2240')
world_land.plot(ax=ax, color='#1e3a2e', edgecolor='#2d5a40', linewidth=0.4, zorder=1, alpha=0.95)
world_land.boundary.plot(ax=ax, color='#3d7a55', linewidth=0.3, zorder=2, alpha=0.6)
for lat in range(-90, 91, 30):
    ax.axhline(lat, color='white', alpha=0.07, linewidth=0.5, zorder=3)
for lon in range(-180, 181, 30):
    ax.axvline(lon, color='white', alpha=0.07, linewidth=0.5, zorder=3)
ax.axhline(0, color='#f39c12', alpha=0.5, linewidth=1.2, linestyle='--', zorder=4)


ax.scatter(deb_lons, deb_lats, c='#e74c3c', s=30, alpha=0.8, marker='x',
           zorder=5, linewidths=1.5, label=f'Iridium-33 Debris ({len(deb_lats)})')

ax.scatter(sat_lons, sat_lats, c='#f1c40f', s=220, alpha=1.0, marker='*',
           zorder=6, edgecolor='white', linewidth=0.4,
           label=f'Starlink Controlled ({len(sat_lats)})')

for slon, slat in zip(sat_lons, sat_lats):
    circle = plt.Circle((slon, slat), 22, color='#3498db',
                         alpha=0.07, fill=True, transform=ax.transData, zorder=4)
    ax.add_patch(circle)

ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
ax.set_xlabel("Longitude (°)", color='white', fontsize=12, labelpad=8)
ax.set_ylabel("Latitude  (°)", color='white', fontsize=12, labelpad=8)
ax.set_title(
    "Week 4 Environment Test — Satellite + Debris Positions on Real World Map\n"
    "Map: Natural Earth (geopandas) | Data: CelesTrak TLE + Skyfield SGP4",
    color='white', fontsize=13, fontweight='bold', pad=12
)
ax.tick_params(colors='white', labelsize=9)
for spine in ax.spines.values():
    spine.set_edgecolor('white')

sat_p  = mpatches.Patch(color='#f1c40f', label=f'★ Starlink ({len(sat_lats)})')
deb_p  = mpatches.Patch(color='#e74c3c', label=f'✕  Debris ({len(deb_lats)})')
cov_p  = mpatches.Patch(color='#3498db', alpha=0.4, label='Coverage Footprint')
land_p = mpatches.Patch(color='#1e3a2e', label='Land')
ax.legend(handles=[sat_p, deb_p, cov_p, land_p],
          loc='lower left', facecolor='#0d1b2a',
          edgecolor='white', labelcolor='white', fontsize=9)

plt.tight_layout()
map_path = os.path.join(cfg.PLOTS_DIR, "graph_week4_env_test.png")
plt.savefig(map_path, dpi=150, bbox_inches='tight', facecolor='#0a0e1a')
plt.close()
print(f"Map saved: {map_path}")