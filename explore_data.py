import pandas as pd
import numpy as np
import os
import config as cfg

starlink_path = os.path.join(cfg.DATA_DIR, "starlink_positions_now.csv")
df_starlink = pd.read_csv(starlink_path)

print(f"   File: {starlink_path}")
print(f"   Shape: {df_starlink.shape[0]} rows × {df_starlink.shape[1]} columns")
print(f"\n   Columns: {list(df_starlink.columns)}")
print(f"\n   First 5 rows:")
print(df_starlink.head().to_string(index=False))

print(f"\n   Statistics:")
print(f"   Altitude: min={df_starlink['altitude_km'].min():.1f} km, "
      f"max={df_starlink['altitude_km'].max():.1f} km, "
      f"avg={df_starlink['altitude_km'].mean():.1f} km")
print(f"   Latitude: min={df_starlink['latitude'].min():.1f}deg, "
      f"max={df_starlink['latitude'].max():.1f}deg")
print(f"   Longitude: min={df_starlink['longitude'].min():.1f}deg, "
      f"max={df_starlink['longitude'].max():.1f}deg")

collision_path = os.path.join(cfg.DATA_DIR, "collision_training_data.csv")
df_collision = pd.read_csv(collision_path)

print(f"   File: {collision_path}")
print(f"   Shape: {df_collision.shape[0]} rows × {df_collision.shape[1]} columns")
print(f"\n   Columns: {list(df_collision.columns)}")
print(f"\n   First 5 closest events (highest risk):")
print(df_collision.head().to_string(index=False))

print(f"\n   Collision Risk Distribution:")
risk_counts = df_collision['collision_risk'].value_counts().sort_index(ascending=False)
for risk, count in risk_counts.items():
    bar = "|" * min(50, int(count / df_collision.shape[0] * 50))
    print(f"   Risk {risk:.2f}: {count:5d} events {bar}")

# How many are in danger zone?
danger_count = (df_collision['distance_km'] < cfg.SAFE_DISTANCE_KM).sum()
warning_count = ((df_collision['distance_km'] >= cfg.SAFE_DISTANCE_KM) &
                 (df_collision['distance_km'] < cfg.DANGER_ZONE_KM)).sum()
safe_count = (df_collision['distance_km'] >= cfg.DANGER_ZONE_KM).sum()


print(f"\n   Distance Zone Summary:")
print(f"   DANGER  (<{cfg.SAFE_DISTANCE_KM} km) : {danger_count:5d} events")
print(f"   WARNING (<{cfg.DANGER_ZONE_KM} km) : {warning_count:5d} events")
print(f"   SAFE    (>{cfg.DANGER_ZONE_KM} km) : {safe_count:5d} events")


