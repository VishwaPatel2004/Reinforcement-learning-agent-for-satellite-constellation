import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import geopandas as gpd
import geodatasets  
import config as cfg

df_s = pd.read_csv(os.path.join(cfg.DATA_DIR, "starlink_positions_now.csv"))
df_c = pd.read_csv(os.path.join(cfg.DATA_DIR, "collision_training_data.csv"))

print (" Data Load Suceessfully! ")

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#f8f9fa')
ax.set_facecolor('#ffffff')
ax.hist(df_s['altitude_km'], bins=15, color='#3498db',
        edgecolor='white', linewidth=1.2, alpha=0.85)

ax.axvline(x=df_s['altitude_km'].mean(), color='#e74c3c',
           linestyle='--', linewidth=2, label=f"Mean altitude: {df_s['altitude_km'].mean():.1f} km")
ax.axvline(x=cfg.SAFE_DISTANCE_KM + df_s['altitude_km'].min(),
           color='orange', linestyle=':', linewidth=2)
ax.set_xlabel("Altitude Above Earth Surface (km)", fontsize=13)
ax.set_ylabel("Number of Satellites", fontsize=13)
ax.set_title("Starlink Satellite Altitude Distribution\n"
             "(Real-time data from CelesTrak — Downloaded using Skyfield)",
             fontsize=14, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

stats_text = (f"Total Satellites: {len(df_s)}\n"
              f"Min Altitude: {df_s['altitude_km'].min():.1f} km\n"
              f"Max Altitude: {df_s['altitude_km'].max():.1f} km\n"
              f"Mean Altitude: {df_s['altitude_km'].mean():.1f} km")
ax.text(0.02, 0.97, stats_text, transform=ax.transAxes,
        verticalalignment='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
plt.tight_layout()
path1 = os.path.join(cfg.PLOTS_DIR, "graph1_altitude_distribution.png")
plt.savefig(path1, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {path1}")


world_land = gpd.read_file(geodatasets.get_path('naturalearth.land'))
print(f"   Loaded {len(world_land)} land polygons from Natural Earth dataset")

fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor('#0a0e1a')
ax.set_facecolor('#0a2240')

ax.fill_between([-180, 180], -90, 90, color='#0a2240', alpha=1.0, zorder=0)

world_land.plot(
    ax         = ax,
    color      = '#1e3a2e',
    edgecolor  = '#2d5a40',
    linewidth  = 0.4,
    zorder     = 1,
    alpha      = 0.95
)

world_land.boundary.plot(
    ax        = ax,
    color     = '#3d7a55',
    linewidth = 0.3,
    zorder    = 2,
    alpha     = 0.6
)

for lat in range(-90, 91, 30):
    ax.axhline(lat, color='white', alpha=0.08, linewidth=0.5, linestyle='-', zorder=3)
for lon in range(-180, 181, 30):
    ax.axvline(lon, color='white', alpha=0.08, linewidth=0.5, linestyle='-', zorder=3)


ax.axhline(0, color='#f39c12', alpha=0.5, linewidth=1.2, linestyle='--', zorder=4)
ax.text(172, 1.5, 'Equator', color='#f39c12', alpha=0.75, fontsize=8,
        fontweight='bold', zorder=5)

ax.axhline(53,  color='#3498db', alpha=0.4, linewidth=1, linestyle=':', zorder=4)
ax.axhline(-53, color='#3498db', alpha=0.4, linewidth=1, linestyle=':', zorder=4)
ax.text(-178, 54.5, '53deg N limit', color='#3498db', alpha=0.65, fontsize=8, zorder=5)
ax.text(-178, -57,  '53deg S limit', color='#3498db', alpha=0.65, fontsize=8, zorder=5)

scatter = ax.scatter(
    df_s['longitude'],
    df_s['latitude'],
    c         = df_s['altitude_km'],
    cmap      = 'plasma',
    s         = 200,
    alpha     = 1.0,
    zorder    = 6,
    marker    = '*',
    edgecolor = 'white',
    linewidth = 0.4,
    label     = f'Starlink Satellites ({len(df_s)})'
)

cbar = plt.colorbar(scatter, ax=ax, shrink=0.55, pad=0.01, aspect=30)
cbar.set_label('Satellite Altitude (km)', color='white', fontsize=10, labelpad=8)
cbar.ax.yaxis.set_tick_params(color='white', labelsize=9)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
cbar.outline.set_edgecolor('white')
cbar.outline.set_linewidth(0.5)

cities = {
    'New York':   (-74.0,  40.7),
    'London':     ( -0.1,  51.5),
    'Mumbai':     ( 72.8,  19.1),
    'Tokyo':      (139.7,  35.7),
    'Sydney':     (151.2, -33.9),
    'Sao Paulo':  (-46.6, -23.5),
    'Cape Town':  ( 18.4, -33.9),
}
for city, (clon, clat) in cities.items():
    ax.plot(clon, clat, 'o', color='#f1c40f', markersize=4, zorder=7, alpha=0.9)
    ax.text(clon + 2, clat + 1.5, city, color='#f1c40f', fontsize=7,
            alpha=0.85, zorder=8, fontweight='bold')

ax.set_xlabel("Longitude (deg)", color='white', fontsize=12, labelpad=8)
ax.set_ylabel("Latitude  (deg)", color='white', fontsize=12, labelpad=8)
ax.set_title(
    "Real-Time Starlink Satellite Positions on Earth — Geographical World Map\n"
    "Map Data: Natural Earth (geopandas) | Orbital Data: CelesTrak TLE + Skyfield SGP4",
    color='white', fontsize=13, fontweight='bold', pad=12
)
ax.set_xlim(-180, 180)
ax.set_ylim( -90,  90)
ax.set_xticks(range(-180, 181, 30))
ax.set_yticks(range( -90,  91, 30))
ax.tick_params(colors='white', labelsize=9)
for spine in ax.spines.values():
    spine.set_edgecolor('white')
    spine.set_linewidth(0.8)

sat_patch    = mpatches.Patch(color='#f1c40f',   label=f'* Starlink Satellite ({len(df_s)})')
land_patch   = mpatches.Patch(color='#1e3a2e',   label='Land (Natural Earth)')
ocean_patch  = mpatches.Patch(color='#0a2240',   label='Ocean')
eq_line      = plt.Line2D([0],[0], color='#f39c12', lw=1.5, ls='--', label='Equator')
inc_line     = plt.Line2D([0],[0], color='#3498db', lw=1,   ls=':',  label='53deg Inclination Limit')
ax.legend(
    handles    = [sat_patch, land_patch, ocean_patch, eq_line, inc_line],
    loc        = 'lower left',
    facecolor  = '#0d1b2a',
    edgecolor  = 'white',
    labelcolor = 'white',
    fontsize   = 9,
    framealpha = 0.85
)
plt.tight_layout()
path2 = os.path.join(cfg.PLOTS_DIR, "graph2_satellite_world_map.png")
plt.savefig(path2, dpi=150, bbox_inches='tight', facecolor='#0a0e1a')
plt.close()
print(f"Saved: {path2}")



fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#f8f9fa')

ax1 = axes[0]
ax1.set_facecolor('#ffffff')
distances = df_c['distance_km'].clip(upper=1000)
ax1.hist(distances, bins=40, color='#9b59b6', edgecolor='white', linewidth=0.8, alpha=0.85)
ax1.axvline(cfg.SAFE_DISTANCE_KM, color='#e74c3c', linewidth=2.5,
            linestyle='--', label=f'Safe Distance ({cfg.SAFE_DISTANCE_KM} km)')
ax1.axvline(cfg.DANGER_ZONE_KM, color='#f39c12', linewidth=2,
            linestyle=':', label=f'Danger Zone ({cfg.DANGER_ZONE_KM} km)')
ax1.set_xlabel("Distance between Satellite and Debris (km)", fontsize=12)
ax1.set_ylabel("Number of Events", fontsize=12)
ax1.set_title("Satellite–Debris Distance Distribution", fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

ax1.axvspan(0, cfg.SAFE_DISTANCE_KM, alpha=0.15, color='red', label='Danger')
ax1.axvspan(cfg.SAFE_DISTANCE_KM, cfg.DANGER_ZONE_KM, alpha=0.1, color='orange')

ax2 = axes[1]
ax2.set_facecolor('#ffffff')
risk_colors = ['#2ecc71', '#27ae60', '#f39c12', '#e67e22', '#e74c3c', '#c0392b']
risk_values = df_c['collision_risk'].value_counts().sort_index()
bars = ax2.bar(range(len(risk_values)),
               risk_values.values,
               color=risk_colors[:len(risk_values)],
               edgecolor='white', linewidth=1, alpha=0.9)
ax2.set_xticks(range(len(risk_values)))
ax2.set_xticklabels([f'{v:.2f}' for v in risk_values.index], fontsize=11)
ax2.set_xlabel("Collision Risk Probability", fontsize=12)
ax2.set_ylabel("Number of Events", fontsize=12)
ax2.set_title("Collision Risk Probability Distribution", fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
for bar, count in zip(bars, risk_values.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f'{count:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
fig.suptitle("Collision Risk Analysis — Satellite vs Iridium-33 Debris\n"
             "Data Source: CelesTrak NORAD Catalog | Analysis: Python/Pandas",
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
path3 = os.path.join(cfg.PLOTS_DIR, "graph3_collision_risk.png")
plt.savefig(path3, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {path3}")



