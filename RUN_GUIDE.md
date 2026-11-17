# Satellite Collision Avoidance: Full Project Evolution Guide

This guide contains every terminal command from the very beginning of the project (data gathering) to the final live AI dashboard. It shows how the project evolved week by week.

---

## WEEK 1: Data Gathering & Orbital Mechanics
**Goal:** Download real satellite data and verify we can compute positions using Skyfield.

### 1. Download Data
```powershell
python download_data.py
```
**Expected Output:**
```text
--- STARTING CELESTRAK DOWNLOADS ---
Downloading [active]...
  [+] Success! Saved 10,234 objects to active_satellites.json
...
```

### 2. Test Orbital Physics (Skyfield)
```powershell
python skyfield_test.py
```
**Expected Output:**
```text
Testing Skyfield on Iridium-33 debris...
Debris piece: IRIDIUM 33 DEB
Position (km): [ 1234.56, -4321.12,  5678.90 ]
Speed: 7.62 km/s
```

---

## WEEK 2: Basic Reinforcement Learning Environment
**Goal:** Build an OpenAI Gym environment (`env.py`) where an AI can observe satellite and debris positions.

### 3. Test the Environment
```powershell
python test_env.py
```
**Expected Output:**
```text
Testing SatelliteEnv...
Observation space: Box(-inf, inf, (427,), float32)
Action space: Box(-1.0, 1.0, (210,), float32)
Step 1: Reward = 0.95 | Collision Prob = 0.00012
```

---

## WEEK 3: Standard Training (Unsafe PPO)
**Goal:** Train a basic PPO agent. This agent learns to save fuel but *ignores safety constraints*, leading to collisions.

### 4. Train Basic Agent
```powershell
python train_basic.py
```
**Expected Output:**
```text
============================================================
WEEK 3: BASIC PPO TRAINING (No Safety Constraints)
============================================================
...
---------------------------------
| rollout/           |          |
|    ep_rew_mean     | 25.4     |
---------------------------------
```

### 5. Plot Basic Training
```powershell
python step04_plot_training.py
```
**Expected Output:**
```text
Generating training plots...
Saved to plots/training_curve.png
```

---

## WEEK 4: Safe PPO (CMDP - Lagrangian Constraints)
**Goal:** The core contribution. Train a Safe PPO agent that strictly penalizes collision probabilities to guarantee safety.

### 6. Train Safe Agent
```powershell
python train_safe.py
```
**Expected Output:**
```text
============================================================
WEEK 6: SAFE PPO TRAINING (With Lagrangian Constraint)
============================================================
Cost limit  : 0.01
...
STARTING SAFE RL TRAINING...
```

---

## WEEK 5: Proving the AI Works (Testing & Comparison)
**Goal:** Mathematically prove that the Safe PPO agent avoids collisions while the Baseline (No Thrust) and Basic PPO crash.

### 7. Compare Basic vs Safe
```powershell
python step05_safe_vs_unsafe.py
```
**Expected Output:**
```text
Testing Basic PPO... [CRASH]
Testing Safe PPO... [SURVIVED]
```

### 8. Run Mathematical Proof (3 Scenarios)
```powershell
python prove_agent_works.py
```
**Expected Output:**
```text
======================================================================
                         SCENARIO COMPARISON                          
======================================================================
Metric                            No Thrust       Random     Safe PPO
----------------------------------------------------------------------
Survived?                         NO [BOOM]     NO [BOOM]     YES [OK]
Collision at step                        45           52            ?
Min distance to debris (km)             0.2          0.8         17.1
======================================================================
```

---

## WEEK 6: The Final Live Dashboard
**Goal:** Combine everything into an interactive 3D Dash web application. The dashboard uses live TLE data, simulates physics, and the Safe PPO agent automatically applies targeted thrust only to threatened satellites.

### 9. Launch Live Dashboard
```powershell
python live_dashboard.py
```
**Expected Output:**
```text
============================================================
  LIVE DASHBOARD READY
  Open : http://localhost:8050
  Sats : 70
  AI   : SAFE PPO (CMDP)
============================================================
*(Open your browser to see the 3D globe and AI evasive maneuvers in real-time!)*

---

## WEEK 7: Multi-Agent Reinforcement Learning (Self Agent)
**Goal:** Transition from centralized control to a scalable, decentralized Multi-Agent system (Self Agent Architecture). Each satellite acts as an independent onboard agent observing a constant 24-dimensional local vector, avoiding debris AND other satellites.

### 10. Train Multi-Agent System (Self Agent)
```powershell
python train_marl.py
```
**Expected Output:**
```text
=================================================================
MULTI-AGENT REINFORCEMENT LEARNING TRAINING (Self Agent Architecture)
=================================================================
Number of Autonomous Satellites : 70
Local Observation Dimension     : 24 (O(1) invariant)
Iter [24/24] | EnvSteps:  6144 | Rew: 1.00 | DebMin: 19.6km | SatMin: 16.0km | Pc: 0.00000 | Lambda: 0.000 | FPS: 5215
MARL TRAINING COMPLETE!
Model saved to: B:\satellite\models\marl_mappo\mappo_final_model.pt
```

### 11. Run Scientific Comparative Benchmark (Centralized vs Self Agent)
```powershell
python step07_compare_marl_vs_centralized.py
```
**Expected Output:**
```text
=================================================================
EVALUATION RESULTS SUMMARY:
=================================================================
Metric                           | Centralized Safe PPO | Self Agent (MARL)
---------------------------------------------------------------------------
Debris Collision Rate            | 0.0                % | 0.0                %
Avg Min Debris Dist              | 38.1              km | 33.2              km
Inter-Satellite Min Dist         | 12.4              km | 37.5              km
Avg Fuel Remaining               | 0.0                % | 99.9               %
Inference Latency / Sat          | 0.007             ms | 0.010             ms
Max Constellation Scalability    | 70                   | 500                 +
=================================================================
[+] Saved plot: B:\satellite\plots\graph8_marl_vs_centralized.png
```

### 12. Run Live Dashboard with Self Agent Enabled
```powershell
python live_dashboard.py
```
*(Select **`🛰️ Self Agent`** in the dropdown to observe real-time decentralized evasive maneuvers!)*

