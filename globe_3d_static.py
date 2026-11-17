import json
import os
import numpy as np
import plotly.graph_objects as go
from skyfield.api import load, EarthSatellite
from datetime import datetime
import datetime as dt
import config as cfg

print("=" * 60)
print("3D INTERACTIVE EARTH GLOBE")
print("Generating satellite visualization...")
print("=" * 60)

ts = load.timescale()
t  = ts.now()

with open(os.path.join(cfg.DATA_DIR, "starlink_satellites.json")) as f:
    starlink_data = json.load(f)[:cfg.N_SATELLITES]

with open(os.path.join(cfg.DATA_DIR, "iridium33_debris.json")) as f:
    debris_data = json.load(f)[:cfg.MAX_DEBRIS]

print(f"  Loaded {len(starlink_data)} Starlink satellites")
print(f"  Loaded {len(debris_data)} Iridium-33 debris pieces")


sat_lats, sat_lons, sat_alts, sat_names, sat_speeds = [], [], [], [], []

for obj in starlink_data:
    try:
        sat = EarthSatellite.from_omm(ts, obj)
        geo = sat.at(t)
        sub = geo.subpoint()
        vel = geo.velocity.km_per_s
        speed = float(np.linalg.norm(vel))
        sat_lats.append(float(sub.latitude.degrees))
        sat_lons.append(float(sub.longitude.degrees))
        sat_alts.append(float(sub.elevation.km))
        sat_names.append(obj.get("OBJECT_NAME", "Unknown"))
        sat_speeds.append(round(speed, 2))
    except Exception:
        continue

deb_lats, deb_lons, deb_alts, deb_names = [], [], [], []

for obj in debris_data:
    try:
        sat = EarthSatellite.from_omm(ts, obj)
        geo = sat.at(t)
        sub = geo.subpoint()
        deb_lats.append(float(sub.latitude.degrees))
        deb_lons.append(float(sub.longitude.degrees))
        deb_alts.append(float(sub.elevation.km))
        deb_names.append(obj.get("OBJECT_NAME", "DEBRIS"))
    except Exception:
        continue

print(f"  Computed positions: {len(sat_lats)} satellites, {len(deb_lats)} debris")


print("  Computing orbit paths (90-minute tracks)...")

def get_orbit_path(sat_obj, ts, n_minutes=90, n_points=180):
    """Get lat/lon path for one satellite over n_minutes."""
    lats, lons = [], []
    base = dt.datetime.utcnow()
    for i in range(n_points):
        future = base + dt.timedelta(seconds=(n_minutes * 60 / n_points) * i)
        tf = ts.utc(future.year, future.month, future.day,
                    future.hour, future.minute, future.second)
        try:
            sub = sat_obj.at(tf).subpoint()
            lats.append(float(sub.latitude.degrees))
            lons.append(float(sub.longitude.degrees))
        except Exception:
            lats.append(None)
            lons.append(None)
    return lats, lons

orbit_tracks = []
for i, obj in enumerate(starlink_data[:3]):
    try:
        sat = EarthSatellite.from_omm(ts, obj)
        lats, lons = get_orbit_path(sat, ts)
        orbit_tracks.append({
            "name": obj.get("OBJECT_NAME", f"SAT-{i}"),
            "lats": lats,
            "lons": lons
        })
    except Exception:
        continue

print(f"  Computed {len(orbit_tracks)} orbit paths")


print("  Computing closest approaches...")

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi  = np.radians(lat2 - lat1)
    dlam  = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

danger_events = []
for si, (slat, slon, sname) in enumerate(zip(sat_lats, sat_lons, sat_names)):
    for di, (dlat, dlon, dname) in enumerate(zip(deb_lats, deb_lons, deb_names)):
        dist = haversine_km(slat, slon, dlat, dlon)
        if dist < 500:
            danger_events.append({
                "sat": sname, "debris": dname,
                "dist_km": round(dist, 1),
                "sat_lat": slat, "sat_lon": slon,
                "deb_lat": dlat, "deb_lon": dlon
            })

danger_events.sort(key=lambda x: x["dist_km"])
top_dangers = danger_events[:5]
print(f"  Found {len(danger_events)} close approaches (<500 km)")



print("\n  Building 3D globe figure...")

ORBIT_COLORS = ["#00d4ff", "#ff6b35", "#7bed9f"]
traces = []

for i, track in enumerate(orbit_tracks):
    color = ORBIT_COLORS[i % len(ORBIT_COLORS)]
    traces.append(go.Scattergeo(
        lat  = track["lats"],
        lon  = track["lons"],
        mode = "lines",
        line = dict(width=1.8, color=color, dash="dot"),
        name = f"🛸 Orbit: {track['name'][:15]}",
        opacity   = 0.6,
        hoverinfo = "name",
        showlegend = True,
    ))

traces.append(go.Scattergeo(
    lat  = deb_lats,
    lon  = deb_lons,
    mode = "markers",
    marker = dict(
        size   = 6,
        color  = "#ff2244",
        symbol = "x",
        line   = dict(width=1.5, color="#ff2244"),
        opacity = 0.9,
    ),
    name = f"☠️  Iridium-33 Debris ({len(deb_lats)} pieces)",
    text = [f"<b>{n}</b><br>Alt: {a:.0f} km<br>Lat: {la:.1f}° Lon: {lo:.1f}°"
            for n, a, la, lo in zip(deb_names, deb_alts, deb_lats, deb_lons)],
    hovertemplate = "%{text}<extra></extra>",
    showlegend = True,
))

traces.append(go.Scattergeo(
    lat  = sat_lats,
    lon  = sat_lons,
    mode = "markers+text",
    marker = dict(
        size   = 14,
        color  = sat_alts,
        colorscale = "Plasma",
        cmin   = 500,
        cmax   = 600,
        symbol = "star",
        line   = dict(width=1, color="white"),
        colorbar = dict(
            title    = dict(text="Altitude (km)", font=dict(color="white", size=11)),
            tickfont = dict(color="white"),
            x        = 1.02,
            thickness = 12,
        ),
    ),
    text  = [n.replace("STARLINK-", "SL-") for n in sat_names],
    textposition = "top right",
    textfont = dict(size=8, color="#FFD700"),
    name  = f"⭐ Starlink Satellites ({len(sat_lats)})",
    customdata = list(zip(sat_names, sat_alts, sat_speeds, sat_lats, sat_lons)),
    hovertemplate = (
        "<b>%{customdata[0]}</b><br>"
        "Altitude : %{customdata[1]:.1f} km<br>"
        "Speed    : %{customdata[2]} km/s<br>"
        "Latitude : %{customdata[3]:.2f}°<br>"
        "Longitude: %{customdata[4]:.2f}°<extra></extra>"
    ),
    showlegend = True,
))

if top_dangers:
    danger_lats = [e["sat_lat"] for e in top_dangers]
    danger_lons = [e["sat_lon"] for e in top_dangers]
    danger_text = [
        f"⚠️ <b>CLOSE APPROACH</b><br>"
        f"Satellite : {e['sat']}<br>"
        f"Debris    : {e['debris']}<br>"
        f"Distance  : <b>{e['dist_km']} km</b>"
        for e in top_dangers
    ]
    traces.append(go.Scattergeo(
        lat  = danger_lats,
        lon  = danger_lons,
        mode = "markers",
        marker = dict(
            size   = 22,
            color  = "rgba(255, 50, 50, 0.15)",
            line   = dict(width=2.5, color="rgba(255, 80, 80, 0.9)"),
            symbol = "circle",
        ),
        name = f"⚠️ Top {len(top_dangers)} Closest Approaches",
        text = danger_text,
        hovertemplate = "%{text}<extra></extra>",
        showlegend = True,
    ))

now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

layout = go.Layout(
    title = dict(
        text = (
            f"<b>🛰️ Satellite Constellation — Real-Time 3D Globe</b><br>"
            f"<span style='font-size:13px; color:#aaaaaa'>"
            f"Starlink (Controlled) + Iridium-33 Debris | "
            f"CelesTrak TLE + Skyfield SGP4 | {now_str}</span>"
        ),
        x    = 0.5,
        font = dict(color="white", size=18),
    ),
    paper_bgcolor = "#050a14",
    geo = dict(
        projection = dict(
            type     = "orthographic",
            rotation = dict(lon=20, lat=20, roll=0),
        ),
        showland        = True,
        landcolor       = "#1a3a28",
        showocean       = True,
        oceancolor      = "#05183a",
        showcoastlines  = True,
        coastlinecolor  = "#2d7a55",
        coastlinewidth  = 0.8,
        showcountries   = True,
        countrycolor    = "#1a4a35",
        countrywidth    = 0.5,
        showframe       = True,
        framecolor      = "#00d4ff",
        framewidth      = 1.5,
        bgcolor         = "#050a14",
        lataxis = dict(showgrid=True, gridcolor="rgba(0,212,255,0.15)",
                       gridwidth=0.8, dtick=30),
        lonaxis = dict(showgrid=True, gridcolor="rgba(0,212,255,0.15)",
                       gridwidth=0.8, dtick=30),
    ),
    legend = dict(
        bgcolor     = "rgba(5, 15, 35, 0.85)",
        bordercolor = "#00d4ff",
        borderwidth = 1,
        font        = dict(color="white", size=11),
        x = 0.01, y = 0.99,
    ),
    margin = dict(l=0, r=0, t=80, b=20),
    annotations = [dict(
        text = (
            "<b>HOW TO USE:</b>   Click & drag = Rotate globe   |   "
            "Scroll = Zoom   |   Hover = See details   |   "
            "Click legend = Show/hide layers"
        ),
        xref = "paper", yref = "paper",
        x = 0.5, y = -0.02,
        showarrow = False,
        font = dict(color="#aaaaaa", size=10),
        align = "center",
    )],
  
    updatemenus = [dict(
        type = "buttons",
        showactive = False,
        x = 0.05, y = 0.05,
        xanchor = "left", yanchor = "bottom",
        bgcolor = "#0a1a2e",
        bordercolor = "#00d4ff",
        font = dict(color="white"),
        buttons = [
            dict(
                label  = "▶  Auto-Rotate",
                method = "animate",
                args   = [None, {"frame": {"duration": 50, "redraw": True},
                                 "fromcurrent": True,
                                 "transition": {"duration": 0}}]
            ),
            dict(
                label  = "⏹  Stop",
                method = "animate",
                args   = [[None], {"frame": {"duration": 0, "redraw": False},
                                   "mode": "immediate",
                                   "transition": {"duration": 0}}]
            ),
        ],
    )],
)

frames = []
for lon in range(0, 360, 3):
    frames.append(go.Frame(
        layout = dict(geo=dict(projection=dict(rotation=dict(lon=lon, lat=20, roll=0)))),
        name   = str(lon)
    ))

fig = go.Figure(data=traces, layout=layout, frames=frames)

out_path = os.path.join(cfg.PLOTS_DIR, "3d_globe_satellite_map.html")
fig.write_html(
    out_path,
    include_plotlyjs = "cdn",
    full_html        = True,
    config = {
        "scrollZoom"    : True,
        "displayModeBar": True,
        "displaylogo"   : False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    }
)

print(f"\n✅ 3D Globe saved: {out_path}")
print("\nFEATURES:")
print("  → Rotate  : Click and drag the globe")
print("  → Zoom    : Scroll wheel")
print("  → Info    : Hover over any satellite or debris dot")
print("  → Auto    : Click '▶ Auto-Rotate' button")
print(f"\nSatellites shown : {len(sat_lats)}")
print(f"Debris shown     : {len(deb_lats)}")
print(f"Orbit paths      : {len(orbit_tracks)} satellites (90-min each)")
print(f"Danger events    : {len(danger_events)} close approaches detected")
if top_dangers:
    print(f"\nTOP 3 CLOSEST APPROACHES RIGHT NOW:")
    for i, e in enumerate(top_dangers[:3], 1):
        status = "⚠️ DANGER" if e["dist_km"] < cfg.SAFE_DISTANCE_KM else "⚠️ WARNING"
        print(f"  {i}. {status} | {e['sat']} ↔ {e['debris']} → {e['dist_km']} km")
print("=" * 60)
print("\nOpen this file in your browser:")
print(f"  {out_path}")