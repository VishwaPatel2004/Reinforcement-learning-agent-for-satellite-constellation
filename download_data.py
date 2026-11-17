import os
import json
import requests
import pandas as pd
import numpy as np
import math
from datetime import datetime
from skyfield.api import load, EarthSatellite


DATA_DIR = r"B:\satellite\data"
os.makedirs(DATA_DIR, exist_ok=True)

CELESTRAK_GROUPS = {
    "active": "active_satellites.json",
    "starlink": "starlink_satellites.json",
    "iridium-33-debris": "iridium33_debris.json",
    "cosmos-2251-debris": "cosmos2251_debris.json",
    "fengyun-1c-debris": "fengyun1c_debris.json",
    "cosmos-1408-debris": "cosmos1408_debris.json"
}

def download_celestrak_data():
    print("--- STARTING CELESTRAK DOWNLOADS ---")
    downloaded_files = {}
    for group, filename in CELESTRAK_GROUPS.items():
        save_path = os.path.join(DATA_DIR, filename)
        downloaded_files[group] = save_path
        
        if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
            print(f"Skipping [{group}] — already downloaded.")
            continue
            
        url = f"https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=json"
        print(f"Downloading [{group}]...")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            print(f"  [+] Success! Saved {len(data):,} objects to {filename}")
        except requests.exceptions.RequestException as e:
            print(f"  [-] Failed to download {group}. Error: {e}")
    print("--- DOWNLOADING COMPLETE ---\n")
    return downloaded_files

def process_coverage_data(starlink_json_path, n_samples=50):
    print("--- PROCESSING COVERAGE DATA (Real Positions) ---")
    if not os.path.exists(starlink_json_path): return
    with open(starlink_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    ts = load.timescale()
    t = ts.now()
    positions = []
    
    for obj in data[:n_samples]:
        try:
            name = obj.get('OBJECT_NAME', 'Unknown')
            sat = EarthSatellite.from_omm(ts, obj)
            subpoint = sat.at(t).subpoint()
            positions.append({
                'name': name,
                'norad_id': obj.get('OBJECT_ID', ''),
                'latitude': subpoint.latitude.degrees,
                'longitude': subpoint.longitude.degrees,
                'altitude_km': subpoint.elevation.km
            })
        except: continue
        
    df = pd.DataFrame(positions)
    df.to_csv(os.path.join(DATA_DIR, "starlink_positions_now.csv"), index=False)
    print(f"  [+] Saved {len(df)} satellites to starlink_positions_now.csv\n")

def process_collision_data(active_path, debris_paths, max_sat=100, max_deb=200):
    print("--- PROCESSING COLLISION DATA (Pairwise Distances) ---")
    ts = load.timescale()
    t = ts.now()
    active_sats, debris_list = [], []
    
    try:
        with open(active_path, 'r', encoding='utf-8') as f:
            for obj in json.load(f)[:max_sat]:
                try:
                    sat = EarthSatellite.from_omm(ts, obj)
                    pos = sat.at(t).position.km
                    active_sats.append({'name': obj['OBJECT_NAME'], 'x': pos[0], 'y': pos[1], 'z': pos[2]})
                except: continue
    except: return

    for d_path in debris_paths:
        try:
            with open(d_path, 'r', encoding='utf-8') as f:
                for obj in json.load(f)[:max_deb]:
                    try:
                        sat = EarthSatellite.from_omm(ts, obj)
                        pos = sat.at(t).position.km
                        debris_list.append({'name': obj['OBJECT_NAME'], 'x': pos[0], 'y': pos[1], 'z': pos[2]})
                    except: continue
        except: pass

    records = []
    print(f"  Calculating {len(active_sats)*len(debris_list)} pairs...")
    for s in active_sats:
        for d in debris_list:
            dist = math.sqrt((s['x']-d['x'])**2 + (s['y']-d['y'])**2 + (s['z']-d['z'])**2)
            
            if dist < 10: risk = 0.99
            elif dist < 50: risk = 0.75
            elif dist < 200: risk = 0.25
            elif dist < 500: risk = 0.05
            else: risk = 0.00
            
            if dist < 8000:
                records.append({
                    'sat_name': s['name'], 'sat_x': round(s['x'], 2), 'sat_y': round(s['y'], 2), 'sat_z': round(s['z'],2),
                    'deb_name': d['name'], 'deb_x': round(d['x'], 2), 'deb_y': round(d['y'], 2), 'deb_z': round(d['z'],2),
                    'distance_km': round(dist, 2), 'collision_risk': risk
                })
                
    # Inject synthetic high-risk data so DANGER/WARNING events exist
    import random
    if len(active_sats) > 0 and len(debris_list) > 0:
        for i in range(100):  # Increased to 100 synthetic events
            s = random.choice(active_sats)
            d = random.choice(debris_list)
            # Make distances extremely close: 50 events < 1.5 km (Near misses), 50 events < 10 km
            dist = random.uniform(0.05, 1.5) if i < 50 else random.uniform(2.0, 9.9)
            risk = 0.99 if dist < 10 else 0.75
            records.append({
                'sat_name': s['name'] + " (SYNTH)", 'sat_x': s['x'], 'sat_y': s['y'], 'sat_z': s['z'],
                'deb_name': d['name'] + " (SYNTH)", 'deb_x': s['x']+dist, 'deb_y': s['y'], 'deb_z': s['z'],
                'distance_km': round(dist, 2), 'collision_risk': risk
            })

    if len(records) > 0:
        df = pd.DataFrame(records).sort_values(by='distance_km', ascending=True)
        df.to_csv(os.path.join(DATA_DIR, "collision_training_data.csv"), index=False)
        print(f"  [+] Saved {len(df)} proximity events.\n")
    else:
        print("  [-] No proximity events found within checking radius.\n")

if __name__ == "__main__":
    downloaded = download_celestrak_data()
    if "starlink" in downloaded: process_coverage_data(downloaded["starlink"])
    if "active" in downloaded:
        debris_paths = [v for k,v in downloaded.items() if "debris" in k]
        process_collision_data(downloaded["active"], debris_paths)
    print("ALL DONE!")
