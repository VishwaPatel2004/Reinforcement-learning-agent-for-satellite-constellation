import json
import os
from skyfield.api import load, EarthSatellite
from datetime import datetime
import datetime as dt
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
import geodatasets
import config as cfg

ts = load.timescale()

starlink_path = os.path.join(cfg.DATA_DIR, "starlink_satellites.json")
with open(starlink_path, 'r') as f:
    all_starlink = json.load(f)

selected = all_starlink[:5]
print(f"\n[LIVE POSITIONS] Calculated at: {datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")

print(f"{'Satellite Name':<30} {'Lat (deg)':>8} {'Lon (deg)':>8} {'Alt (km)':>10} {'Speed (km/s)':>14}")

t_now = ts.now()
satellites = []
sat_lats, sat_lons, sat_alts, sat_names = [], [], [], []

for obj in selected:
    try:
        sat = EarthSatellite.from_omm(ts, obj)
        satellites.append(sat)
        geocentric = sat.at(t_now)
        subpoint   = geocentric.subpoint()
        velocity   = geocentric.velocity.km_per_s
        speed      = (velocity[0]**2 + velocity[1]**2 + velocity[2]**2) ** 0.5
        name = obj.get('OBJECT_NAME', 'Unknown')[:28]
        sat_lats.append(subpoint.latitude.degrees)
        sat_lons.append(subpoint.longitude.degrees)
        sat_alts.append(subpoint.elevation.km)
        sat_names.append(name)
        print(f"{name:<30} {subpoint.latitude.degrees:>8.2f} "
              f"{subpoint.longitude.degrees:>8.2f} "
              f"{subpoint.elevation.km:>10.1f} "
              f"{speed:>14.3f}")
    except Exception as e:
        print(f"Skipped: {e}")

print("\n" + "-" * 60)
print("[FUTURE PREDICTION] Starlink-1 position over the next 30 minutes:")

print(f"{'Time (UTC)':<20} {'Lat (deg)':>8} {'Lon (deg)':>10} {'Alt (km)':>10}")


base_time   = dt.datetime.now(dt.timezone.utc)
future_lats = []
future_lons = []
future_mins = [0, 5, 10, 15, 20, 25, 30]
for minutes_ahead in future_mins:
    future_time = base_time + dt.timedelta(minutes=minutes_ahead)
    t_future = ts.utc(future_time.year, future_time.month, future_time.day,
                      future_time.hour, future_time.minute, future_time.second)
    geocentric = satellites[0].at(t_future)
    subpoint   = geocentric.subpoint()
    future_lats.append(subpoint.latitude.degrees)
    future_lons.append(subpoint.longitude.degrees)
    time_str = future_time.strftime('%H:%M:%S UTC')
    print(f"{time_str:<20} {subpoint.latitude.degrees:>8.2f} "
          f"{subpoint.longitude.degrees:>10.2f} "
          f"{subpoint.elevation.km:>10.1f}")

print("\n[MAP] Creating real-world satellite position map...")
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
ax.text(172, 1.5, 'Equator', color='#f39c12', fontsize=8, fontweight='bold', zorder=5)
# 30-minute future track of satellite 1 (dashed trail)
ax.plot(future_lons, future_lats,
        color='#e74c3c', linewidth=2.5, linestyle='--',
        zorder=5, alpha=0.85, label='30-min future track (STARLINK-1)')

for i, (flon, flat, fmin) in enumerate(zip(future_lons, future_lats, future_mins)):
    ax.plot(flon, flat, 'o', color='#e74c3c', markersize=5, zorder=6, alpha=0.9)
    if i % 2 == 0:
        ax.text(flon + 1.5, flat + 1.5, f'+{fmin}min',
                color='#e74c3c', fontsize=7, fontweight='bold', zorder=7)
        

for i, (slon, slat, salt, sname) in enumerate(zip(sat_lons, sat_lats, sat_alts, sat_names)):
    ax.scatter(slon, slat, c='#f1c40f', s=250, marker='*',
               zorder=8, edgecolor='white', linewidth=0.5)
    ax.text(slon + 2, slat + 2, sname.replace('STARLINK-', 'SL-'),
            color='#f1c40f', fontsize=7, fontweight='bold', zorder=9)
ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
ax.set_xlabel("Longitude (deg)", color='white', fontsize=12, labelpad=8)
ax.set_ylabel("Latitude  (deg)", color='white', fontsize=12, labelpad=8)
ax.set_title(
    f"Live Starlink Satellite Positions - {datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC\n"
    "Map: Natural Earth (geopandas) | Orbit: CelesTrak TLE + Skyfield SGP4 Propagation",
    color='white', fontsize=13, fontweight='bold', pad=12
)
ax.tick_params(colors='white', labelsize=9)
for spine in ax.spines.values():
    spine.set_edgecolor('white')
sat_patch   = mpatches.Patch(color='#f1c40f', label=f'* Current Position ({len(sat_lats)} satellites)')
trk_line    = plt.Line2D([0],[0], color='#e74c3c', lw=2, ls='--', label='30-min Future Track')
land_patch  = mpatches.Patch(color='#1e3a2e', label='Land (Natural Earth)')
ocean_patch = mpatches.Patch(color='#0a2240', label='Ocean')
ax.legend(handles=[sat_patch, trk_line, land_patch, ocean_patch],
          loc='lower left', facecolor='#0d1b2a', edgecolor='white',
          labelcolor='white', fontsize=9, framealpha=0.85)
plt.tight_layout()
map_path = os.path.join(cfg.PLOTS_DIR, "graph_step03_live_positions.png")
plt.savefig(map_path, dpi=150, bbox_inches='tight', facecolor='#0a0e1a')
plt.close()
print(f"Map saved: {map_path}")


# UPDATED MAP CODE — All 5 Satellite Future Tracks on Real World Map

print("\n[MAP] Creating real-world map with ALL 5 satellite future tracks...")

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
ax.text(172, 1.5, 'Equator', color='#f39c12', fontsize=8, fontweight='bold', zorder=5)

SAT_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']


future_mins = [0, 5, 10, 15, 20, 25, 30]


for sat_idx, (sat, slon, slat, salt, sname) in enumerate(
        zip(satellites, sat_lons, sat_lats, sat_alts, sat_names)):

    color = SAT_COLORS[sat_idx % len(SAT_COLORS)]
    short_name = sname.replace('STARLINK-', 'SL-')

    f_lats = []
    f_lons = []

    for minutes_ahead in future_mins:
        future_time = base_time + dt.timedelta(minutes=minutes_ahead)
        t_future = ts.utc(
            future_time.year, future_time.month, future_time.day,
            future_time.hour, future_time.minute, future_time.second
        )
        geocentric = sat.at(t_future)
        subpoint   = geocentric.subpoint()
        f_lats.append(subpoint.latitude.degrees)
        f_lons.append(subpoint.longitude.degrees)

    ax.plot(f_lons, f_lats,
            color=color, linewidth=2.0, linestyle='--',
            zorder=5, alpha=0.80)

    for i, (flon, flat, fmin) in enumerate(zip(f_lons, f_lats, future_mins)):
        ax.plot(flon, flat, 'o', color=color, markersize=4, zorder=6, alpha=0.9)
        if i % 2 == 0:
            ax.text(flon + 1.5, flat + 1.5, f'+{fmin}m',
                    color=color, fontsize=6, fontweight='bold', zorder=7)

    ax.scatter(slon, slat, c=color, s=280, marker='*',
               zorder=8, edgecolor='white', linewidth=0.6)

    ax.text(slon + 2, slat + 2, short_name,
            color=color, fontsize=7, fontweight='bold', zorder=9,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#0a0e1a',
                      alpha=0.6, edgecolor=color, linewidth=0.5))

ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
ax.set_xlabel("Longitude (deg)", color='white', fontsize=12, labelpad=8)
ax.set_ylabel("Latitude  (deg)", color='white', fontsize=12, labelpad=8)
ax.set_title(
    f"Live Starlink Positions + 30-min Future Tracks - {datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC\n"
    "All 5 Satellites | Map: Natural Earth (geopandas) | Orbit: CelesTrak TLE + Skyfield SGP4",
    color='white', fontsize=13, fontweight='bold', pad=12
)
ax.tick_params(colors='white', labelsize=9)
for spine in ax.spines.values():
    spine.set_edgecolor('white')

legend_handles = []
for sat_idx, sname in enumerate(sat_names):
    short = sname.replace('STARLINK-', 'SL-')
    color = SAT_COLORS[sat_idx % len(SAT_COLORS)]
    handle = plt.Line2D([0], [0], color=color, lw=2, ls='--',
                        marker='*', markersize=10,
                        label=f'{short} + 30-min track')
    legend_handles.append(handle)

legend_handles.append(mpatches.Patch(color='#1e3a2e', label='Land (Natural Earth)'))
legend_handles.append(mpatches.Patch(color='#0a2240', label='Ocean'))

ax.legend(handles=legend_handles,
          loc='lower left', facecolor='#0d1b2a',
          edgecolor='white', labelcolor='white',
          fontsize=8, framealpha=0.9)

plt.tight_layout()
map_path = os.path.join(cfg.PLOTS_DIR, "graph_step03_all5_future_tracks.png")
plt.savefig(map_path, dpi=150, bbox_inches='tight', facecolor='#0a0e1a')
plt.close()
print(f"Map saved: {map_path}")


