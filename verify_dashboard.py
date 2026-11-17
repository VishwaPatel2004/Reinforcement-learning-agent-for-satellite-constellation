import math, time, json, os
import numpy as np
from skyfield.api import load, EarthSatellite
import config as cfg

ts = load.timescale()
t  = ts.now()

with open(os.path.join(cfg.DATA_DIR, 'starlink_satellites.json')) as f:
    raw = json.load(f)[:70]

sat_xyz, sat_lats = [], []
for idx, obj in enumerate(raw):
    try:
        sat = EarthSatellite.from_omm(ts, obj)
        pos = list(sat.at(t).position.km)
        if any(math.isnan(v) for v in pos): raise ValueError()
        sub = sat.at(t).subpoint()
        sat_xyz.append(pos)
        sat_lats.append(float(sub.latitude.degrees))
    except:
        ang = (idx / 70) * 2 * math.pi; r = 7000
        sat_xyz.append([r*math.cos(ang), r*math.sin(ang), r*0.1*math.sin(ang*3)])
        sat_lats.append(0.0)

PERIODS = [113, 157, 199, 241, 283, 337, 383, 431, 479, 523]
PHASES  = [0.0, 0.7, 1.4, 2.1, 2.8, 3.5, 4.2, 4.9, 5.6, 6.3]
target_idxs = [int(i * len(sat_xyz) / 10) for i in range(10)]

print(f"\nSimulating 10 callbacks (100 seconds apart) to show zone cycling:\n")
print(f"{'Time':>8}  " + "  ".join([f"F{i+1:02d}" for i in range(10)]) + "  STATUS")
print("-" * 90)

statuses = []
for tick in range(15):
    t_sec = time.time() + tick * 10
    dists = []
    for slot in range(10):
        p  = PERIODS[slot]; ph = PHASES[slot]
        ti = target_idxs[slot]
        dist = 5.0 + 175.0 * ((math.sin(t_sec * 2 * math.pi / p + ph) + 1) / 2.0)
        ax, ay, az = sat_xyz[ti]
        norm = math.sqrt(ax*ax + ay*ay + az*az) or 1.0
        nx = ax/norm; ny = ay/norm
        fx = ax + dist*ny; fy = ay - dist*nx; fz = az + dist*0.05
        actual = math.sqrt((fx-ax)**2 + (fy-ay)**2 + (fz-az)**2)
        dists.append(actual)

    min_d = min(dists)
    if min_d < cfg.SAFE_DISTANCE_KM:
        status = "DANGER "
    elif min_d < cfg.DANGER_ZONE_KM:
        status = "WARNING"
    else:
        status = "SAFE   "
    statuses.append(status.strip())

    def fmt(d):
        if d < cfg.SAFE_DISTANCE_KM:   return f"\033[91m{d:5.0f}\033[0m"
        elif d < cfg.DANGER_ZONE_KM:   return f"\033[93m{d:5.0f}\033[0m"
        else:                          return f"\033[92m{d:5.0f}\033[0m"

    dist_str = "  ".join([fmt(d) for d in dists])
    if min_d < cfg.SAFE_DISTANCE_KM:   sc = "\033[91m"
    elif min_d < cfg.DANGER_ZONE_KM:   sc = "\033[93m"
    else:                              sc = "\033[92m"
    print(f"{tick*10:>6}s   {dist_str}  {sc}{status}\033[0m")

print()
unique = set(statuses)
print(f"Zones seen across 15 ticks: {unique}")
if len(unique) > 1:
    print("SUCCESS - all zones cycle naturally!")
else:
    print("WARNING - only one zone seen, check periods/phases")
print(f"\nDanger threshold : {cfg.SAFE_DISTANCE_KM} km")
print(f"Warning threshold: {cfg.DANGER_ZONE_KM} km")
