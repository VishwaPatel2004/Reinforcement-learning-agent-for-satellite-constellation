"""
Diagnostic: runs the exact same get_positions() and min_distance_numpy()
as live_dashboard.py and prints exactly what the dashboard will show.
"""
import json, os, math, time
import numpy as np
from skyfield.api import load, EarthSatellite
import config as cfg

ts = load.timescale()
t  = ts.now()

DISPLAY_N_SATS = 70
DISPLAY_N_DEB  = 150

# ── Load satellites ──────────────────────────────────────────
with open(os.path.join(cfg.DATA_DIR, "starlink_satellites.json")) as f:
    starlink_raw = json.load(f)[:DISPLAY_N_SATS]

debris_raw = []
for fname, label in [
    ("iridium33_debris.json",  "Iridium-33"),
    ("cosmos2251_debris.json", "Cosmos-2251"),
    ("fengyun1c_debris.json",  "Fengyun-1C"),
    ("cosmos1408_debris.json", "Cosmos-1408"),
]:
    path = os.path.join(cfg.DATA_DIR, fname)
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        debris_raw.extend(data)
        print(f"  Loaded {label}: {len(data)} pieces")

debris_raw = debris_raw[:DISPLAY_N_DEB]

sat_objects, sat_names = [], []
for obj in starlink_raw:
    try:
        sat_objects.append(EarthSatellite.from_omm(ts, obj))
        sat_names.append(obj.get("OBJECT_NAME", "Starlink"))
    except: continue

deb_objects, deb_names_master = [], []
for obj in debris_raw:
    try:
        deb_objects.append(EarthSatellite.from_omm(ts, obj))
        deb_names_master.append(obj.get("OBJECT_NAME", "DEBRIS"))
    except: continue

print(f"\nSatellites built: {len(sat_objects)}")
print(f"Debris built    : {len(deb_objects)}")

# ── Compute real positions ──────────────────────────────────
sat_xyz, sat_lats, sat_alts = [], [], []
failed_sats = 0
for idx, sat in enumerate(sat_objects):
    try:
        geo = sat.at(t)
        sub = geo.subpoint()
        pos = list(geo.position.km)
        if any(math.isnan(v) for v in pos): raise ValueError()
        sat_xyz.append(pos)
        sat_lats.append(float(sub.latitude.degrees))
        sat_alts.append(float(sub.elevation.km))
    except:
        failed_sats += 1
        ang = (idx / max(len(sat_objects), 1)) * 2 * math.pi
        r = 7000.0
        sat_xyz.append([r*math.cos(ang), r*math.sin(ang), r*0.1*math.sin(ang*3)])
        sat_lats.append(0.0); sat_alts.append(550.0)

real_deb_xyz, real_deb_alts, real_deb_names = [], [], []
failed_deb = 0
for deb, name in zip(deb_objects, deb_names_master):
    try:
        geo = deb.at(t)
        sub = geo.subpoint()
        pos = list(geo.position.km)
        if any(math.isnan(v) for v in pos): raise ValueError()
        real_deb_xyz.append(pos)
        real_deb_alts.append(float(sub.elevation.km))
        real_deb_names.append(name)
    except:
        failed_deb += 1

print(f"Failed sat SGP4 : {failed_sats}")
print(f"Failed deb SGP4 : {failed_deb}")
print(f"Valid real debris: {len(real_deb_xyz)}")

# ── Check real debris vs satellites ──────────────────────────
print("\n--- REAL DEBRIS vs SATELLITES (closest 10 pairs) ---")
s = np.array(sat_xyz)
d = np.array(real_deb_xyz)
diff  = s[:, np.newaxis, :] - d[np.newaxis, :, :]
dists = np.sqrt((diff**2).sum(axis=2))

# Find top 10 closest real pairs
flat = dists.flatten()
sorted_idx = np.argsort(flat)[:10]
for rank, fi in enumerate(sorted_idx):
    si, di = divmod(fi, dists.shape[1])
    print(f"  #{rank+1:2d}  {sat_names[si][:25]:25s}  <->  {real_deb_names[di][:25]:25s}  = {dists[si,di]:.1f} km  (deb alt={real_deb_alts[di]:.0f} km)")

# ── Check synthetic fragment math ─────────────────────────────
print("\n--- SYNTHETIC FRAGMENT DISTANCES (at startup) ---")
T0 = time.time()
ACTIVE_PERIODS = [480, 780, 1260]
target_dists_km = [5.0, 35.0, 120.0]
ACTIVE_PHASES = []
for d_target, p in zip(target_dists_km, ACTIVE_PERIODS):
    sin_t = (d_target - 5.0) / 87.5 - 1.0
    sin_t = max(-1.0, min(1.0, sin_t))
    nat   = T0 * 2 * math.pi / p
    ACTIVE_PHASES.append(math.asin(sin_t) - nat)

n = len(sat_xyz)
target_idxs = [int(i * n / 10) for i in range(3)]
synth_xyz = []
for slot in range(3):
    p = ACTIVE_PERIODS[slot]; ph = ACTIVE_PHASES[slot]
    dist = 5.0 + 175.0 * ((math.sin(T0 * 2 * math.pi / p + ph) + 1) / 2.0)
    ti = target_idxs[slot]
    ax, ay, az = sat_xyz[ti]
    norm = math.sqrt(ax*ax + ay*ay + az*az) or 1.0
    nx = ax/norm; ny = ay/norm
    fpos = [ax + dist*ny, ay - dist*nx, az + dist*0.05]
    synth_xyz.append(fpos)
    # actual 3D dist
    actual = math.sqrt((fpos[0]-ax)**2 + (fpos[1]-ay)**2 + (fpos[2]-az)**2)
    zone = "DANGER" if actual<cfg.SAFE_DISTANCE_KM else ("WARNING" if actual<cfg.DANGER_ZONE_KM else "SAFE")
    print(f"  Active-{slot+1}: intended={dist:.1f}km  actual_3D={actual:.1f}km  target_sat={sat_names[ti][:20]}  [{zone}]")

# ── Full min distance (real + synthetic) ──────────────────────
all_deb_xyz = real_deb_xyz + synth_xyz
s2 = np.array(sat_xyz)
d2 = np.array(all_deb_xyz)
diff2 = s2[:, np.newaxis, :] - d2[np.newaxis, :, :]
dists2 = np.sqrt((diff2**2).sum(axis=2))
idx2 = np.unravel_index(np.argmin(dists2), dists2.shape)
min_d = float(dists2[idx2])
si2, di2 = int(idx2[0]), int(idx2[1])
all_deb_names = real_deb_names + [f"ACTIVE-{i+1}" for i in range(3)]

is_synth = "SYNTHETIC" if di2 >= len(real_deb_xyz) else "REAL"
zone_final = "DANGER" if min_d<cfg.SAFE_DISTANCE_KM else ("WARNING" if min_d<cfg.DANGER_ZONE_KM else "SAFE")
print(f"\n--- DASHBOARD WILL SHOW ---")
print(f"  MIN DIST  : {min_d:.1f} km  ({is_synth} pair)")
print(f"  Satellite : {sat_names[si2]}")
print(f"  Debris    : {all_deb_names[di2] if di2 < len(all_deb_names) else 'SYNTH'}")
print(f"  STATUS    : {zone_final}")
print(f"\n  DANGER threshold : {cfg.SAFE_DISTANCE_KM} km")
print(f"  WARNING threshold: {cfg.DANGER_ZONE_KM} km")
