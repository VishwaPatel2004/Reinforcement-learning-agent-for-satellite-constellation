# =============================================================================
# step06_final_report_graphs.py
# WEEK 7: Generate all remaining graphs for the final report
#
# Produces:
#   graph7_earth_map_coverage.png  ← Real geopandas world map
#   graph8_episode_detail.png
# =============================================================================

import os, json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
import geodatasets
from skyfield.api import load, EarthSatellite
from safe_env import SafeSatelliteEnv
from stable_baselines3 import PPO
import config as cfg

print("=" * 60)
print("WEEK 7: GENERATING FINAL REPORT GRAPHS")
print("=" * 60)

# ═════════════════════════════════════════════════════════════════════════════
# GRAPH 7: Coverage Footprints + Iridium Debris on REAL WORLD MAP
# ═════════════════════════════════════════════════════════════════════════════
print("[1/2] Creating real-world Earth coverage map...")

ts = load.timescale()
t  = ts.now()

# Load satellite and debris TLE data
with open(os.path.join(cfg.DATA_DIR, "starlink_satellites.json")) as f:
    slink = json.load(f)[:cfg.N_SATELLITES]
with open(os.path.join(cfg.DATA_DIR, "iridium33_debris.json")) as f:
    ideb  = json.load(f)[:cfg.MAX_DEBRIS]

sat_lats, sat_lons = [], []
for obj in slink:
    try:
        s   = EarthSatellite.from_omm(ts, obj)
        sub = s.at(t).subpoint()
        sat_lats.append(sub.latitude.degrees)
        sat_lons.append(sub.longitude.degrees)
    except: continue

deb_lats, deb_lons = [], []
for obj in ideb:
    try:
        d   = EarthSatellite.from_omm(ts, obj)
        sub = d.at(t).subpoint()
        deb_lats.append(sub.latitude.degrees)
        deb_lons.append(sub.longitude.degrees)
    except: continue

# Load real world map
print("   Loading Natural Earth map...")
world_land = gpd.read_file(geodatasets.get_path('naturalearth.land'))

# --- RECONSTRUCTED MISSING CODE ---
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_facecolor('#0a2240')
world_land.plot(ax=ax, color='#1e3a2e', edgecolor='#2d5a40', linewidth=0.5, zorder=1, alpha=0.95)
world_land.boundary.plot(ax=ax, color='#3d7a55', linewidth=0.4, zorder=2, alpha=0.6)

for lat in range(-90, 91, 30): ax.axhline(lat, color='white', alpha=0.08, linewidth=0.5, zorder=3)
for lon in range(-180, 181, 30): ax.axvline(lon, color='white', alpha=0.08, linewidth=0.5, zorder=3)
ax.axhline(0, color='#f39c12', alpha=0.5, linewidth=1, linestyle='--', zorder=4)
# -------------------------------------

# Iridium-33 debris danger zone rings (10 km → degrees)
DANGER_DEG = cfg.SAFE_DISTANCE_KM / 111.0
for dlon, dlat in zip(deb_lons[:20], deb_lats[:20]):
    ring = plt.Circle((dlon, dlat), DANGER_DEG * 15,  # enlarge for visibility
                       color='#e74c3c', alpha=0.06, fill=True,
                       transform=ax.transData, zorder=4)
    ax.add_patch(ring)

# Coverage footprints for each satellite (22° half-angle)
for slon, slat in zip(sat_lons, sat_lats):
    circle = plt.Circle((slon, slat), 22, color='#3498db',
                         alpha=0.09, fill=True, transform=ax.transData, zorder=5)
    ax.add_patch(circle)

# Plot debris (red X)
ax.scatter(deb_lons, deb_lats, c='#e74c3c', s=35, alpha=0.9, marker='x',
           zorder=6, linewidths=1.5, label=f'Iridium-33 Debris ({len(deb_lats)} pieces)')

# Plot satellites (yellow star)
ax.scatter(sat_lons, sat_lats, c='#f1c40f', s=220, alpha=1.0, marker='*',
           zorder=7, edgecolor='white', linewidth=0.5,
           label=f'Starlink Satellites ({len(sat_lats)} controlled)')

ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
ax.set_xlabel("Longitude (°)", color='white', fontsize=12, labelpad=8)
ax.set_ylabel("Latitude  (°)", color='white', fontsize=12, labelpad=8)
ax.set_title(
    "Satellite Coverage Footprints and Iridium-33 Debris Field — Real World Map\n"
    "Map: Natural Earth (geopandas) | Data: CelesTrak TLE + Skyfield SGP4 Propagation",
    color='white', fontsize=14, fontweight='bold', pad=12
)
ax.set_xticks(range(-180, 181, 30))
ax.set_yticks(range(-90, 91, 30))
ax.tick_params(colors='white', labelsize=9)
for spine in ax.spines.values():
    spine.set_edgecolor('white')

sat_p  = mpatches.Patch(color='#f1c40f',           label=f'★ Starlink ({len(sat_lats)})')
deb_p  = mpatches.Patch(color='#e74c3c',           label=f'✕ Iridium-33 Debris ({len(deb_lats)})')
cov_p  = mpatches.Patch(color='#3498db', alpha=0.4, label='Coverage Zone (22°)')
dng_p  = mpatches.Patch(color='#e74c3c', alpha=0.3, label='Debris Danger Zone')
land_p = mpatches.Patch(color='#1e3a2e',           label='Land (Natural Earth)')
ocn_p  = mpatches.Patch(color='#0a2240',           label='Ocean')

ax.legend(handles=[sat_p, deb_p, cov_p, dng_p, land_p, ocn_p],
          loc='lower left', facecolor='#0d1b2a',
          edgecolor='white', labelcolor='white', fontsize=9, framealpha=0.9)

plt.tight_layout()
path7 = os.path.join(cfg.PLOTS_DIR, "graph7_earth_map_coverage.png")
plt.savefig(path7, dpi=150, bbox_inches='tight', facecolor='#0a0e1a')
plt.close()
print(f"   ✅ Saved: {path7}")

# ═════════════════════════════════════════════════════════════════════════════
# GRAPH 8: Episode Detail — What Happens Over One Orbit
# ═════════════════════════════════════════════════════════════════════════════
print("[2/2] Creating episode detail graph...")

# Graceful loading in case training stopped early
try:
    with open(os.path.join(cfg.MODELS_DIR, "latest_safe_run.txt")) as f:
        safe_path = f.read().strip().split('\n')[1]
except FileNotFoundError:
    safe_path = None

env   = SafeSatelliteEnv(render_mode=None)
if safe_path:
    model = PPO.load(safe_path)
else:
    model = None
obs, _ = env.reset(seed=999)

steps_data = {"step":[], "coverage":[], "min_dist":[], "collision_prob":[], "lambda":[]}
while True:
    if model:
        action, _ = model.predict(obs, deterministic=True)
    else:
        action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    steps_data["step"].append(info['step'])
    steps_data["coverage"].append(info['coverage'] * 100)
    steps_data["min_dist"].append(info['min_distance_km'])
    steps_data["collision_prob"].append(info['collision_prob'])
    steps_data["lambda"].append(info['lambda_cost'])
    if term or trunc: break

fig, axes = plt.subplots(4, 1, figsize=(13, 14), sharex=True)
fig.patch.set_facecolor('#f8f9fa')
fig.suptitle("Safe PPO Agent — One Complete Episode (90 minutes / 1 orbit)\n"
             "Showing coverage, safety, and Lagrangian constraint evolution",
             fontsize=14, fontweight='bold')

time_min = [s * cfg.TIMESTEP_SECONDS / 60 for s in steps_data["step"]]

# Coverage
axes[0].plot(time_min, steps_data["coverage"], color='#2ecc71', lw=2)
axes[0].set_ylabel("Earth Coverage (%)", fontsize=11)
axes[0].set_title("Earth Coverage by Satellite Constellation", fontsize=12)
axes[0].fill_between(time_min, steps_data["coverage"], alpha=0.2, color='#2ecc71')
axes[0].grid(True, alpha=0.3)

# Min distance
axes[1].plot(time_min, steps_data["min_dist"], color='#3498db', lw=2)
axes[1].axhline(cfg.SAFE_DISTANCE_KM, color='#e74c3c', lw=2, ls='--',
                label=f'Safe distance ({cfg.SAFE_DISTANCE_KM} km)')
axes[1].axhline(cfg.DANGER_ZONE_KM, color='#f39c12', lw=1.5, ls=':',
                label=f'Danger zone ({cfg.DANGER_ZONE_KM} km)')
axes[1].set_ylabel("Min Dist to Debris (km)", fontsize=11)
axes[1].set_title("Minimum Satellite–Debris Separation Distance", fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

# Collision probability
axes[2].plot(time_min, steps_data["collision_prob"], color='#e74c3c', lw=2)
axes[2].axhline(cfg.COST_LIMIT, color='navy', lw=2, ls='--',
                label=f'Constraint limit d={cfg.COST_LIMIT}')
axes[2].set_ylabel("Collision Probability P_c", fontsize=11)
axes[2].set_title("Collision Probability (Gaussian Model, σ=5km)", fontsize=12)
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)
axes[2].fill_between(time_min, steps_data["collision_prob"], cfg.COST_LIMIT,
                     where=[c > cfg.COST_LIMIT for c in steps_data["collision_prob"]],
                     alpha=0.3, color='red', label='Constraint violated')

# Lambda
axes[3].plot(time_min, steps_data["lambda"], color='#9b59b6', lw=2)
axes[3].set_ylabel("Lagrange Multiplier λ", fontsize=11)
axes[3].set_title("Safety Constraint: Lagrange Multiplier λ (increases when unsafe)", fontsize=12)
axes[3].set_xlabel("Simulation Time (minutes)", fontsize=12)
axes[3].grid(True, alpha=0.3)
axes[3].fill_between(time_min, steps_data["lambda"], alpha=0.2, color='#9b59b6')

plt.tight_layout()
path8 = os.path.join(cfg.PLOTS_DIR, "graph8_episode_detail.png")
plt.savefig(path8, dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✅ Saved: {path8}")

print("\n" + "=" * 60)
print("WEEK 7 GRAPHS DONE!")
print("\nYou now have ALL 8 graphs:")
for i in range(1, 9):
    graphs = {1:"altitude_distribution", 2:"satellite_world_map",
              3:"collision_risk", 4:"learning_curve", 5:"random_vs_trained",
              6:"safe_vs_unsafe", 7:"earth_map_coverage", 8:"episode_detail"}
    fn = graphs.get(i, "")
    path = os.path.join(cfg.PLOTS_DIR, f"graph{i}_{fn}.png")
    exists = "✅" if os.path.exists(path) else "⚠️  not found"
    print(f"  graph{i}: {fn}.png — {exists}")
print("=" * 60)
