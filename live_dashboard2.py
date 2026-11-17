# =============================================================================
#
# What this does:
#   → Opens a web dashboard at  http://localhost:8050
#   → 3D rotating Earth globe with LIVE satellite positions
#   → Auto-refreshes every 10 seconds with new real positions
#   → If trained model exists  → shows AI agent's THRUST DECISIONS
#   → If no model yet          → runs in "Live Tracking Only" mode
#   → Shows danger alerts, coverage %, AI status card
#
# CONNECTS TO:
#   data/starlink_satellites.json   ← Week 1 download
#   data/iridium33_debris.json      ← Week 1 download
#   models/safe_ppo_final.zip       ← Week 6 training (if done)
#   models/basic_ppo_final.zip      ← Week 5 training (if done)
#   config.py                       ← All settings
#
# Run:  python live_dashboard.py
# Open: http://localhost:8050
# Stop: Ctrl+C
# =============================================================================

import json, os
import numpy as np
from datetime import datetime
import datetime as dt

from skyfield.api import load, EarthSatellite

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go

import config as cfg

# ── STEP 1: Load TLE data ONCE at startup ────────────────────────────────────
print("=" * 60)
print("LIVE SATELLITE TRACKING DASHBOARD  (RL-Connected)")
print("=" * 60)
print("\n[1/3] Loading satellite TLE data...")

_ts = load.timescale()
_start_time = dt.datetime.utcnow()

with open(os.path.join(cfg.DATA_DIR, "starlink_satellites.json")) as f:
    _starlink_raw = json.load(f)[:cfg.N_SATELLITES]

with open(os.path.join(cfg.DATA_DIR, "iridium33_debris.json")) as f:
    _debris_raw = json.load(f)[:cfg.MAX_DEBRIS]

_sat_objects, _sat_names = [], []
for obj in _starlink_raw:
    try:
        _sat_objects.append(EarthSatellite.from_omm(_ts, obj))
        _sat_names.append(obj.get("OBJECT_NAME", "Unknown"))
    except Exception:
        continue

_deb_objects, _deb_names = [], []
for obj in _debris_raw:
    try:
        _deb_objects.append(EarthSatellite.from_omm(_ts, obj))
        _deb_names.append(obj.get("OBJECT_NAME", "DEBRIS"))
    except Exception:
        continue

print(f"    Satellites : {len(_sat_objects)}")
print(f"    Debris     : {len(_deb_objects)}")

# ── STEP 2: Try to load trained RL models ────────────────────────────────────
print("\n[2/3] Checking for trained RL models...")

_safe_model  = None
_basic_model = None
_ai_mode     = "LIVE TRACKING ONLY"   # default if no model found

try:
    from stable_baselines3 import PPO

    safe_path = os.path.join(cfg.MODELS_DIR, "safe_ppo_final.zip")
    if os.path.exists(safe_path):
        _safe_model = PPO.load(safe_path)
        _ai_mode = "SAFE PPO (CMDP)"
        print(f"    ✅ Safe PPO loaded   : {safe_path}")
    else:
        print(f"    ⚠️  Safe PPO not found (complete Week 6 first)")

    basic_path = os.path.join(cfg.MODELS_DIR, "basic_ppo_final.zip")
    if os.path.exists(basic_path):
        _basic_model = PPO.load(basic_path)
        print(f"    ✅ Basic PPO loaded  : {basic_path}")
    else:
        print(f"    ⚠️  Basic PPO not found (complete Week 5 first)")

    if _safe_model is None and _basic_model is not None:
        _safe_model = _basic_model
        _ai_mode = "BASIC PPO (No Safety)"

except ImportError:
    print("    ⚠️  stable-baselines3 not available")

if _safe_model is None:
    print("    ℹ️  No model — running in Live Tracking mode")
    print("       Complete Week 5/6 training to enable AI mode")
else:
    print(f"    🤖 AI Mode active: {_ai_mode}")

print("\n[3/3] Starting server...")

# ── HELPER FUNCTIONS ─────────────────────────────────────────────────────────
def get_positions_now(n_intervals=0):
    """Get lat/lon/alt/speed + velocity for all satellites and debris."""
    # Fast-forward 100 seconds for every 1 interval tick
    simulated_time = _start_time + dt.timedelta(seconds=(2 * n_intervals))
    t = _ts.utc(simulated_time.year, simulated_time.month, simulated_time.day, 
                simulated_time.hour, simulated_time.minute, simulated_time.second)
    sat_lats, sat_lons, sat_alts, sat_speeds = [], [], [], []
    sat_vx, sat_vy, sat_vz = [], [], []

    for sat in _sat_objects:
        try:
            geo = sat.at(t)
            sub = geo.subpoint()
            vel = geo.velocity.km_per_s
            sat_lats.append(float(sub.latitude.degrees))
            sat_lons.append(float(sub.longitude.degrees))
            sat_alts.append(float(sub.elevation.km))
            sat_speeds.append(round(float(np.linalg.norm(vel)), 2))
            sat_vx.append(float(vel[0]))
            sat_vy.append(float(vel[1]))
            sat_vz.append(float(vel[2]))
        except Exception:
            sat_lats.append(0.0);  sat_lons.append(0.0)
            sat_alts.append(550.0); sat_speeds.append(7.5)
            sat_vx.append(0.0);    sat_vy.append(7.5); sat_vz.append(0.0)

    deb_lats, deb_lons, deb_alts = [], [], []
    for deb in _deb_objects:
        try:
            geo = deb.at(t)
            sub = geo.subpoint()
            deb_lats.append(float(sub.latitude.degrees))
            deb_lons.append(float(sub.longitude.degrees))
            deb_alts.append(float(sub.elevation.km))
        except Exception:
            deb_lats.append(0.0); deb_lons.append(0.0); deb_alts.append(800.0)

    return (sat_lats, sat_lons, sat_alts, sat_speeds,
            sat_vx, sat_vy, sat_vz,
            deb_lats, deb_lons, deb_alts)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin(np.radians(lat2 - lat1) / 2) ** 2 +
         np.cos(p1) * np.cos(p2) *
         np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


def compute_coverage(sat_lats, sat_lons):
    RADIUS = 15
    covered, total = 0, 360
    for tlon in range(-180, 180, 2):
        for tlat in range(-90, 90, 5):
            for slat, slon in zip(sat_lats, sat_lons):
                if abs(tlat - slat) < RADIUS and abs(tlon - slon) < RADIUS:
                    covered += 1
                    break
    return round(covered / total * 100, 1)


def build_observation(sat_lats, sat_lons, sat_alts,
                      sat_vx, sat_vy, sat_vz,
                      deb_lats, deb_lons, deb_alts):
    """
    Build the 67-dimensional state vector from real positions.
    Matches exactly what env.py uses:
      [0:60]  = 10 satellites × 6 values (x,y,z,vx,vy,vz)
      [60:63] = closest debris position (x,y,z)
      [63]    = normalised min distance
      [64]    = collision probability P_c
      [65]    = coverage fraction
      [66]    = fuel (always 1.0 in live mode)
    """
    R   = cfg.EARTH_RADIUS_KM
    obs = np.zeros(67, dtype=np.float32)

    for i, (lat, lon, alt, vx, vy, vz) in enumerate(
            zip(sat_lats, sat_lons, sat_alts, sat_vx, sat_vy, sat_vz)):
        if i >= 10:
            break
        r   = R + alt
        phi = np.radians(lat)
        lam = np.radians(lon)
        obs[i*6+0] = r * np.cos(phi) * np.cos(lam) / (R + 600)
        obs[i*6+1] = r * np.cos(phi) * np.sin(lam) / (R + 600)
        obs[i*6+2] = r * np.sin(phi)               / (R + 600)
        obs[i*6+3] = vx / 8.0
        obs[i*6+4] = vy / 8.0
        obs[i*6+5] = vz / 8.0

    min_dist = 99999.0
    closest_idx = 0
    for si, (slat, slon) in enumerate(zip(sat_lats, sat_lons)):
        for di, (dlat, dlon) in enumerate(zip(deb_lats, deb_lons)):
            d = haversine_km(slat, slon, dlat, dlon)
            if d < min_dist:
                min_dist    = d
                closest_idx = di

    if deb_lats:
        r   = R + deb_alts[closest_idx]
        phi = np.radians(deb_lats[closest_idx])
        lam = np.radians(deb_lons[closest_idx])
        obs[60] = r * np.cos(phi) * np.cos(lam) / (R + 600)
        obs[61] = r * np.cos(phi) * np.sin(lam) / (R + 600)
        obs[62] = r * np.sin(phi)               / (R + 600)

    obs[63] = min(min_dist / 1000.0, 1.0)
    obs[64] = float(np.exp(-((min_dist / 5.0) ** 2)))
    obs[65] = compute_coverage(sat_lats, sat_lons) / 100.0
    obs[66] = 1.0
    return obs, min_dist


def get_ai_actions(obs):
    """
    Ask the trained PPO model what thrust each satellite should apply.
    Returns list of arrow dicts: {dlat, dlon, magnitude}
    """
    if _safe_model is None:
        return None
    try:
        action, _ = _safe_model.predict(obs, deterministic=True)
        arrows = []
        for i in range(min(len(_sat_objects), 10)):
            dvx = float(action[i * 3 + 0])
            dvy = float(action[i * 3 + 1])
            dvz = float(action[i * 3 + 2])
            mag = np.sqrt(dvx**2 + dvy**2 + dvz**2)
            # Convert thrust vector to visible lat/lon direction on globe
            arrows.append({
                "dlat"     : dvy * 8.0,   # scale for globe visibility
                "dlon"     : dvx * 8.0,
                "magnitude": round(mag, 4),
            })
        return arrows
    except Exception:
        return None


# ── BUILD GLOBE ───────────────────────────────────────────────────────────────
def build_globe(sat_lats, sat_lons, sat_alts, sat_speeds,
                sat_vx, sat_vy, sat_vz,
                deb_lats, deb_lons, deb_alts,
                rotation_lon=30, ai_arrows=None):

    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    traces  = []

    # 1. Debris (red X)
    traces.append(go.Scattergeo(
        lat=deb_lats, lon=deb_lons, mode="markers",
        marker=dict(size=7, color="#ff1744", symbol="x",
                    line=dict(width=1.8, color="#ff1744"), opacity=0.9),
        name=f"☠️  Iridium-33 Debris ({len(deb_lats)})",
        text=[f"<b>{n}</b><br>Alt: {a:.0f} km"
              for n, a in zip(_deb_names, deb_alts)],
        hovertemplate="%{text}<extra></extra>",
    ))

    # 2. Find closest pair
    min_dist = 99999.0
    closest_sat, closest_deb = "—", "—"
    min_s, min_d = None, None
    for si, (slat, slon) in enumerate(zip(sat_lats, sat_lons)):
        for di, (dlat, dlon) in enumerate(zip(deb_lats, deb_lons)):
            d = haversine_km(slat, slon, dlat, dlon)
            if d < min_dist:
                min_dist    = d
                closest_sat = _sat_names[si] if si < len(_sat_names) else "SAT"
                closest_deb = _deb_names[di] if di < len(_deb_names) else "DEB"
                min_s, min_d = (slat, slon), (dlat, dlon)

    # 3. Danger line
    if min_dist < 300 and min_s and min_d:
        lc = "#ff0000" if min_dist < cfg.SAFE_DISTANCE_KM else "#ff9800"
        traces.append(go.Scattergeo(
            lat=[min_s[0], min_d[0]], lon=[min_s[1], min_d[1]],
            mode="lines+markers",
            line=dict(width=3, color=lc, dash="dash"),
            marker=dict(size=14, color="rgba(255,50,50,0.1)",
                        line=dict(width=2, color=lc)),
            name=f"⚠️ Closest: {min_dist:.0f} km",
            hoverinfo="name",
        ))

    # 4. Coverage circles
    for slat, slon in zip(sat_lats, sat_lons):
        cl, cn = [], []
        for angle in range(0, 366, 12):
            cl.append(slat + 12 * np.cos(np.radians(angle)))
            cn.append(slon + 12 * np.sin(np.radians(angle)))
        traces.append(go.Scattergeo(lat=cl, lon=cn, mode="lines",
            line=dict(width=0.5, color="rgba(0,160,255,0.2)"),
            hoverinfo="none", showlegend=False))

    # 5. AI Thrust Arrows (green lines + triangle endpoints)
    if ai_arrows:
        all_lats, all_lons = [], []
        end_lats, end_lons, mags = [], [], []
        for i, (slat, slon, arrow) in enumerate(
                zip(sat_lats, sat_lons, ai_arrows)):
            elat = slat + arrow["dlat"]
            elon = slon + arrow["dlon"]
            all_lats += [slat, elat, None]
            all_lons += [slon, elon, None]
            end_lats.append(elat)
            end_lons.append(elon)
            mags.append(arrow["magnitude"])

        traces.append(go.Scattergeo(lat=all_lats, lon=all_lons, mode="lines",
            line=dict(width=2.5, color="#00ff88"),
            name=f"🤖 AI Thrust ({_ai_mode})", hoverinfo="none"))

        traces.append(go.Scattergeo(lat=end_lats, lon=end_lons, mode="markers",
            marker=dict(size=10, color=mags, colorscale="Greens",
                        symbol="triangle-up", line=dict(width=1, color="white")),
            name="🤖 Thrust Target",
            text=[f"AI: Δv={m:.4f} km/s" for m in mags],
            hovertemplate="%{text}<extra></extra>"))

    # 6. Satellites (yellow stars)
    hover = [f"<b>{n}</b><br>Alt: {a:.1f} km<br>Speed: {s} km/s<br>"
             f"Lat: {la:.2f}° Lon: {lo:.2f}°"
             for n, a, s, la, lo in zip(
                 _sat_names, sat_alts, sat_speeds, sat_lats, sat_lons)]
    traces.append(go.Scattergeo(lat=sat_lats, lon=sat_lons, mode="markers",
        marker=dict(size=16, color=sat_alts, colorscale="Plasma",
                    cmin=500, cmax=600, symbol="star",
                    line=dict(width=1.2, color="white"),
                    colorbar=dict(
                        title=dict(text="Alt (km)",
                                   font=dict(color="white", size=10)),
                        tickfont=dict(color="white", size=9),
                        x=1.01, thickness=10, len=0.5)),
        name=f"⭐ Starlink ({len(sat_lats)})",
        text=hover, hovertemplate="%{text}<extra></extra>"))

    # Layout
    if min_dist < cfg.SAFE_DISTANCE_KM:
        ac, at = "#ff1744", "🔴 DANGER!"
    elif min_dist < cfg.DANGER_ZONE_KM:
        ac, at = "#ff9800", "🟡 WARNING"
    else:
        ac, at = "#00e676", "🟢 SAFE"

    ai_lbl = f"  |  🤖 {_ai_mode}" if _safe_model else "  |  📡 Tracking Only"

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=(f"<b>🛰️  Live Satellite Tracking + AI</b>   "
                  f"<span style='color:#aaa;font-size:12px'>| {now_str}{ai_lbl}</span>   "
                  f"<span style='color:{ac};font-size:13px'>| {at}  {min_dist:.0f} km</span>"),
            x=0.5, font=dict(color="white", size=16)),
        paper_bgcolor="#030810",
        geo=dict(
            projection=dict(type="orthographic",
                            rotation=dict(lon=rotation_lon, lat=15, roll=0)),
            showland=True,    landcolor="#12291a",
            showocean=True,   oceancolor="#020c1e",
            showcoastlines=True, coastlinecolor="#1a5c38", coastlinewidth=0.7,
            showcountries=True,  countrycolor="#1a3a28",
            showframe=True,   framecolor="#00d4ff", framewidth=2,
            bgcolor="#030810",
            lataxis=dict(showgrid=True, gridcolor="rgba(0,212,255,0.10)",
                         gridwidth=0.5, dtick=30),
            lonaxis=dict(showgrid=True, gridcolor="rgba(0,212,255,0.10)",
                         gridwidth=0.5, dtick=30)),
        legend=dict(bgcolor="rgba(3,12,30,0.88)", bordercolor="#00d4ff",
                    borderwidth=1, font=dict(color="white", size=10),
                    x=0.01, y=0.99),
        margin=dict(l=5, r=5, t=65, b=5), height=640)
    return fig, min_dist, closest_sat, closest_deb


# ── DASH APP LAYOUT ───────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="🛰️ Live Satellite Tracker + AI")

C  = {"background": "rgba(0,212,255,0.07)",
      "border": "1px solid rgba(0,212,255,0.3)",
      "borderRadius": "10px", "padding": "15px",
      "textAlign": "center", "flex": "1", "minWidth": "130px"}
CT = {"color":"#aaa","fontSize":"10px","marginBottom":"4px",
      "textTransform":"uppercase","letterSpacing":"1px"}
CV = {"color":"white","fontSize":"22px","fontWeight":"bold","margin":"4px 0"}
CU = {"color":"#aaa","fontSize":"11px"}

app.layout = html.Div(
    style={"backgroundColor":"#030810","minHeight":"100vh",
           "fontFamily":"Inter, Segoe UI, sans-serif","padding":"12px"},
    children=[
        html.Div([
            html.H1("🛰️  Live Satellite Tracking + AI (FAST FORWARD MODE: 10s = 100s)",
                    style={"color":"#00d4ff","fontSize":"20px",
                           "margin":"0","fontWeight":"bold"}),
            html.P(f"Starlink + Iridium-33 | Skyfield SGP4 | "
                   f"AI: {_ai_mode} | Updates every 10 sec (Simulates 100s per tick)",
                   style={"color":"#aaa","margin":"4px 0 0 0","fontSize":"12px"}),
        ], style={"marginBottom":"10px"}),

        html.Div(id="stats-cards",
                 style={"display":"flex","gap":"8px",
                        "flexWrap":"wrap","marginBottom":"10px"}),

        dcc.Graph(id="globe-3d", style={"height":"640px"},
                  config={"scrollZoom":True,"displayModeBar":True,
                          "displaylogo":False}),

        html.Div(id="ai-panel",    style={"marginTop":"8px"}),
        html.Div(id="danger-alert",style={"marginTop":"8px"}),

        html.Div("🎓 DA-IICT | Safe RL for Satellite Collision Avoidance | "
                 "Data: CelesTrak | Physics: Skyfield SGP4 | AI: SB3 PPO",
                 style={"color":"#444","fontSize":"11px",
                        "textAlign":"center","marginTop":"8px"}),

        dcc.Interval(id="interval", interval=10_000, n_intervals=0),
        dcc.Store(id="rotation-store", data=30),
    ])


@app.callback(
    Output("globe-3d",       "figure"),
    Output("stats-cards",    "children"),
    Output("ai-panel",       "children"),
    Output("danger-alert",   "children"),
    Output("rotation-store", "data"),
    Input("interval",        "n_intervals"),
    Input("rotation-store",  "data"),
)
def update_dashboard(n_intervals, rotation_lon):
    new_rotation = (rotation_lon + 5) % 360

    (sat_lats, sat_lons, sat_alts, sat_speeds,
     sat_vx, sat_vy, sat_vz,
     deb_lats, deb_lons, deb_alts) = get_positions_now(n_intervals)

    obs, _ = build_observation(sat_lats, sat_lons, sat_alts,
                               sat_vx, sat_vy, sat_vz,
                               deb_lats, deb_lons, deb_alts)
    ai_arrows = get_ai_actions(obs)

    fig, min_dist, closest_sat, closest_deb = build_globe(
        sat_lats, sat_lons, sat_alts, sat_speeds,
        sat_vx, sat_vy, sat_vz,
        deb_lats, deb_lons, deb_alts,
        rotation_lon=new_rotation, ai_arrows=ai_arrows)

    coverage  = compute_coverage(sat_lats, sat_lons)
    p_collide = float(np.exp(-((min_dist / 5.0) ** 2)))

    if min_dist < cfg.SAFE_DISTANCE_KM:
        sc, status = "#ff1744", "DANGER"
    elif min_dist < cfg.DANGER_ZONE_KM:
        sc, status = "#ff9800", "WARNING"
    else:
        sc, status = "#00e676", "SAFE"

    ai_c = "#00ff88" if _safe_model else "#888"

    cards = [
        html.Div([html.Div("STATUS",  style=CT),
                  html.Div(status, style={**CV,"color":sc,"fontSize":"18px"}),
                  html.Div(datetime.utcnow().strftime('%H:%M:%S UTC'), style=CU)],
                 style={**C,"border":f"1px solid {sc}"}),
        html.Div([html.Div("AI MODE", style=CT),
                  html.Div("ON" if _safe_model else "OFF",
                           style={**CV,"color":ai_c,"fontSize":"18px"}),
                  html.Div(_ai_mode[:16], style=CU)],
                 style={**C,"border":f"1px solid {ai_c}"}),
        html.Div([html.Div("SATELLITES", style=CT),
                  html.Div(len(sat_lats),  style=CV),
                  html.Div("Starlink (LEO)", style=CU)], style=C),
        html.Div([html.Div("DEBRIS",    style=CT),
                  html.Div(len(deb_lats), style=CV),
                  html.Div("Iridium-33",  style=CU)], style=C),
        html.Div([html.Div("MIN DIST",  style=CT),
                  html.Div(f"{min_dist:.1f}", style={**CV,"color":sc}),
                  html.Div("km sat↔debris", style=CU)],
                 style={**C,"border":f"1px solid {sc}"}),
        html.Div([html.Div("P(COLLISION)", style=CT),
                  html.Div(f"{p_collide:.5f}",
                           style={**CV,"color":"#ff9800"
                                  if p_collide > cfg.COST_LIMIT else "white",
                                  "fontSize":"16px"}),
                  html.Div(f"Limit={cfg.COST_LIMIT}", style=CU)], style=C),
        html.Div([html.Div("COVERAGE",    style=CT),
                  html.Div(f"{coverage}%", style=CV),
                  html.Div("Earth footprint", style=CU)], style=C),
        html.Div([html.Div("AVG ALT",   style=CT),
                  html.Div(f"{np.mean(sat_alts):.0f}", style=CV),
                  html.Div("km (LEO)", style=CU)], style=C),
        html.Div([html.Div("AVG SPEED", style=CT),
                  html.Div(f"{np.mean(sat_speeds):.2f}", style=CV),
                  html.Div("km/s",      style=CU)], style=C),
    ]

    if ai_arrows:
        rows = [html.Div([
            html.Span("🤖  AI THRUST DECISIONS — ",
                      style={"color":"#00ff88","fontWeight":"bold","fontSize":"13px"}),
            html.Span(f"Model: {_ai_mode}",
                      style={"color":"#aaa","fontSize":"12px"}),
        ], style={"marginBottom":"6px"})]
        for arrow, sname in zip(ai_arrows, _sat_names):
            if arrow["magnitude"] > 0.001:
                rows.append(html.Div(
                    f"{sname[:15]:15s}  Δv={arrow['magnitude']:.4f} km/s  "
                    f"[N:{arrow['dlat']:+.2f}°  E:{arrow['dlon']:+.2f}°]",
                    style={"color":"#00ff88","fontSize":"11px",
                           "fontFamily":"monospace","margin":"1px 0"}))
        ai_panel = html.Div(rows, style={
            "background":"rgba(0,255,136,0.05)",
            "border":"1px solid rgba(0,255,136,0.3)",
            "borderRadius":"8px", "padding":"10px"})
    else:
        ai_panel = html.Div(
            "🤖  AI MODE: Complete Week 5 or 6 training to see agent thrust decisions here.",
            style={"color":"#555","fontSize":"12px","textAlign":"center",
                   "padding":"10px","border":"1px solid #333","borderRadius":"8px"})

    if min_dist < cfg.DANGER_ZONE_KM:
        bg = "rgba(255,23,68,0.15)" if min_dist < 10 else "rgba(255,152,0,0.15)"
        bd = "#ff1744" if min_dist < 10 else "#ff9800"
        alert = html.Div(
            f"⚠️  CLOSE APPROACH  |  {closest_sat}  ↔  {closest_deb}  |  "
            f"Distance: {min_dist:.1f} km  |  Safe: {cfg.SAFE_DISTANCE_KM:.0f} km",
            style={"background":bg,"border":f"1px solid {bd}",
                   "borderRadius":"8px","padding":"10px","color":bd,
                   "fontWeight":"bold","fontSize":"13px","textAlign":"center"})
    else:
        alert = html.Div(
            f"✅  All systems nominal  |  Min sep: {min_dist:.1f} km  |  "
            f"P(collision): {p_collide:.6f}  |  Limit: {cfg.COST_LIMIT}",
            style={"color":"#00e676","textAlign":"center","fontSize":"12px",
                   "padding":"8px","border":"1px solid #00e67622",
                   "borderRadius":"8px","background":"rgba(0,230,118,0.05)"})

    return fig, cards, ai_panel, alert, new_rotation


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(f"  LIVE DASHBOARD — AI: {_ai_mode}")
    print(f"  Open browser:  http://localhost:8050")
    print(f"  Satellites:    {len(_sat_objects)}")
    print(f"  Debris:        {len(_deb_objects)}")
    print("  Press Ctrl+C to stop")
    print("=" * 60 + "\n")
    app.run(debug=False, host="0.0.0.0", port=8050)