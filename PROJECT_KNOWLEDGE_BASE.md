# Satellite Safe-RL Project Knowledge Base

Last audited: 2026-09-15 (Asia/Calcutta)

Purpose: this is the primary reference for future work on this workspace. Read this file first. Re-open individual source files only when changing them, verifying whether they changed after this audit, or resolving a detail not recorded here.

## 1. Executive summary

This workspace is a Python research prototype for satellite-constellation control with reinforcement learning. It combines:

1. CelesTrak OMM-format orbital records stored as JSON.
2. Skyfield/SGP4 conversion of records to position and velocity at an initial epoch or dashboard refresh time.
3. A custom Gymnasium continuous-control environment.
4. PyTorch vectorization for satellite-debris distances and coverage calculations.
5. Stable-Baselines3 PPO, A2C, DDPG, SAC, and TD3 plus sb3-contrib TRPO.
6. Training logs, saved model archives, evaluation scripts, static plots, interactive Plotly HTML, and Dash dashboards.
7. PDF, PowerPoint, Word, and Markdown presentation artifacts.

The project is a visually strong, runnable academic prototype. It is not currently a valid demonstration that “Safe PPO guarantees collision avoidance.” The largest blockers are saturated coverage/reward, invalid observation bounds, simplified/inconsistent thrust physics, an adaptive penalty shared by both “basic” and “safe” environments, synthetic/hard-coded evaluation results, and stale orbital records.

Recommended defensible description:

> PPO trained in a simplified constellation simulator with an adaptive Lagrangian-style distance-risk penalty, initialized from real orbital-element records.

Do not describe it as flight-grade, formally guaranteed, continuously live, or experimentally superior to every baseline until the validation issues in Section 12 are fixed.

## 2. Verified runtime facts

Audit commands were read-only except that `test_env.py` regenerated `plots/graph_week4_env_test.png`, as that is part of its normal test behavior.

- All Python sources compiled successfully with `python -m compileall -q .`.
- `SatelliteEnv` constructed successfully on CPU.
- State shape: `(427,)`, dtype `float32`.
- Action shape: `(210,)`, range `[-0.01, 0.01]`.
- Stable-Baselines3 `check_env()` failed because reset observations violate `Box(-1, 1)`.
- One checker run found invalid values around `-99.32`, `-54.12`, and `-17.55` at observation indices 153-155.
- The project test run reported observation range `[-51.559, 19.215]` but printed “PASSED” because it asserts shape only, not bounds.
- Full random episode: 540 steps, 2.1 seconds, 254.8 steps/sec on CPU.
- Full random episode return: `539.994` out of a nominal 540 coverage reward.
- Full random episode average coverage: `100.00%`.
- Full random episode minimum distance: `41.52 km`; no collision termination.
- Five initial random steps each returned approximately `1.000` reward and zero collision risk.
- Setting `lambda_cost=5` and calling reset left it at `5.0`; multiplier state leaks across episodes.
- Latest Safe PPO model loaded successfully.
- Loaded Safe PPO model spaces: observation `(427,)`, action `(210,)`.
- Loaded Safe PPO policy parameter count: `77,029`.
- The current system Python has the ML dependencies. The bundled artifact Python has PDF libraries but not `stable_baselines3`.
- The directory is not a Git repository.

## 3. Intended end-to-end workflow

```text
CelesTrak query
    -> OMM JSON snapshots
    -> Skyfield parsing and initial propagation
    -> SatelliteEnv.reset()
    -> 427-value observation
    -> RL policy (PPO or baseline)
    -> 210 continuous actions
    -> simplified environment transition
    -> coverage, distance, risk, fuel, reward
    -> rollout storage
    -> algorithm update
    -> checkpoint/evaluation logs
    -> comparison plots and dashboard
```

Training repeats observation through model update. Dashboard inference loads frozen models and calls `predict(..., deterministic=True)`.

## 4. Configuration (`config.py`, 48 lines)

Import-time behavior:

- Defines four absolute Windows directories under `B:\satellite`.
- Creates data, plot, model, and log directories immediately.
- Prints configuration information on every import.

Configured values:

| Setting | Value | Meaning |
|---|---:|---|
| `N_SATELLITES` | 70 | Controlled satellites |
| `MAX_DEBRIS` | 500 | Maximum debris records used by environment |
| `SAFE_DISTANCE_KM` | 15 | Danger threshold |
| `DANGER_ZONE_KM` | 50 | Warning threshold |
| `EARTH_RADIUS_KM` | 6371 | Spherical Earth radius |
| `TIMESTEP_SECONDS` | 10 | Environment step duration label |
| `EPISODE_LENGTH` | 540 | 90-minute episode |
| `LEARNING_RATE` | 3e-4 | PPO optimizer rate |
| `GAMMA` | 0.99 | Discount factor |
| `N_STEPS` | 2048 | PPO rollout length |
| `BATCH_SIZE` | 64 | PPO mini-batch size |
| `N_EPOCHS` | 10 | PPO epochs per rollout |
| `TOTAL_TIMESTEPS` | 50,000 | Main training budget |
| `COST_LIMIT` | 0.01 | Risk proxy constraint target |
| `LAMBDA_INIT/LR/MAX` | 0 / 0.01 / 10 | Adaptive penalty parameters |
| coverage/danger/fuel reward | 1 / -1 / -0.01 | Reward coefficients |
| `SEED` | 42 | Default training seed |

Issues:

- Absolute paths make the project non-portable.
- Directory creation and console output should not occur merely by importing settings.
- Several documents state `COST_LIMIT=0.001`, but code uses `0.01`.

## 5. Core environment (`env.py`, 287 lines)

### Constructor, lines 11-65

- Selects CUDA if available, otherwise CPU.
- Loads orbital JSON records.
- Creates a Skyfield timescale.
- Creates a 5-degree global latitude/longitude grid: 36 x 72 = 2,592 points.
- Converts grid points to Earth-centered Cartesian vectors and unit-normalizes them.
- Uses a 60-degree angular coverage threshold.
- Declares state dimension `70*6 + 3 + 4 = 427`.
- Declares action dimension `70*3 = 210`.
- Declares observation space `Box(-1,1)` and action space `Box(-0.01,0.01)`.

### Data loading, lines 67-87

- Takes the first 70 Starlink JSON records.
- Concatenates debris in this order: Iridium-33, Cosmos-2251, Fengyun-1C, Cosmos-1408.
- Truncates the combined list to 500. This means later debris families can be partly or completely excluded.
- File errors are not handled; record errors later use bare `except`.

### Reset, lines 89-167

- Seeds NumPy, Python random, and Torch when a seed is supplied.
- Selects a random offset within one day from `ts.now()`.
- Uses Skyfield once to obtain initial position and velocity.
- Bad satellite records are skipped, then missing rows are padded with zero position/velocity.
- Bad debris records are skipped; if all fail, one far-away fallback object is inserted.
- Converts arrays to Torch float32 tensors.
- Sets accumulated action displacement to zero and fuel to 1.
- Calls `_update_physics()` and `_build_state()`.
- Does **not** reset `lambda_cost`; this is an episode-state bug.

### Physics and metrics, lines 169-189

Implemented position equations:

```text
sat_pos = sat_pos0 + sat_vel0 * t + sat_thrust_acc
deb_pos = deb_pos0 + deb_vel0 * t
```

- No gravity, orbit curvature, perturbations, covariance, or repeated SGP4 propagation.
- `torch.cdist` builds the 70 x debris-count distance matrix.
- Minimum distance is the global minimum across every pair.
- Risk proxy is `exp(-distance^2/(2*5^2))`; maximum pairwise value is reported as “collision probability.”
- Coverage unit-normalizes satellite and ground vectors, then counts a grid point covered if any satellite direction is within 60 degrees.

Scientific/implementation issues:

- The risk value is an uncalibrated distance score, not physical collision probability.
- A 60-degree Earth-centered criterion ignores horizon geometry, Earth occlusion, elevation masks, antenna footprints, and altitude.
- With 70 satellites the criterion saturates at 100%, eliminating the coverage learning signal.
- Linear motion over 90 minutes is not a valid orbital trajectory.

### Step, lines 191-241

- Clips action.
- Reshapes it to 70 x 3.
- Adds `action * 10 seconds` to `sat_thrust_acc`.
- Fuel decrement is `sum(abs(action))*0.001`.
- Updates the simplified physics.
- Starts reward at coverage fraction.
- Adds a linear penalty only inside 15 km.
- Adds a tiny fuel penalty.
- Updates lambda per step using `risk - cost_limit`.
- Subtracts `lambda * risk`.
- Terminates under 1 km and subtracts 1,000.
- Truncates at 540 steps or zero fuel.

Key issues:

- The variable named acceleration is a cumulative position offset; control units are inconsistent.
- Delta-v should update velocity and then be integrated consistently.
- The base `SatelliteEnv` already contains the Lagrangian penalty, so it is not an unsafe/basic environment.
- Penalty uses instantaneous risk, while documentation often describes episode-average constrained optimization.
- Fuel depletion is unlikely at the current scale and does not model propellant physically.

### State builder, lines 243-262

- Positions divided by 10,000 and original velocities by 10.
- Chooses one globally closest debris coordinate, not closest debris per satellite.
- Appends normalized minimum distance, risk, coverage, and fuel.
- Returns a CPU NumPy array.

Key issues:

- Actual positions can exceed declared normalized bounds, especially with stale TLE propagation.
- The closest debris coordinate is not paired with the identity/location of the threatened satellite.
- Controlled velocity changes are not represented; state always uses `sat_vel0_t`.
- Bad records padded at Earth center are physically invalid.

## 6. Safe environment (`safe_env.py`, 69 lines)

- Subclasses `SatelliteEnv`.
- Clears `episode_costs` on reset.
- Calls base step without modifying policy optimization.
- Treats reported risk as cost and stores it.
- Adds `cost`, running average, and `constraint_ok` to `info`.
- Provides mean, maximum, violation count, and constraint summary.

Important interpretation:

- This class is a metric wrapper; it does not implement a separate constrained PPO optimizer.
- Standard SB3 PPO sees only the already-penalized scalar reward.
- The base class and safe class share the same safety-shaped reward.
- Calling the method “CMDP with guarantee” is unsupported.

## 7. Training code

### `train_basic.py` (101 lines)

- Creates timestamped model and log directories.
- Uses `DummyVecEnv(Monitor(SatelliteEnv()))` for training and evaluation.
- PPO settings include GAE 0.95, clip 0.2, entropy coefficient 0.01, plus config values.
- Evaluates every 10,000 steps over five deterministic episodes.
- Checkpoints at 50,000 steps.
- Saves `final_model.zip` and `models/latest_basic_run.txt`.

Current artifact issue: `models/latest_basic_run.txt` is missing even though a basic-model directory exists. Scripts depending on the pointer fall back incorrectly.

### `train_safe.py` (105 lines)

- Uses the same SB3 PPO class with `SafeSatelliteEnv`.
- `SafetyLogCallback` logs risk, average episode risk, lambda, and constraint status.
- Evaluates every 10,000 steps over five episodes.
- Saves checkpoints and `safe_final_model.zip`.
- Writes `models/latest_safe_run.txt`.

Issues:

- Callback appends every running-average value, not only final episode costs.
- `ep_rewards` is declared but unused.
- Eval environment has its own persistent lambda across episodes.
- Best model is selected by ordinary reward, not by a lexicographic safety constraint.
- Because the base environment already applies lambda, the safe/basic distinction is not real.

### Individual baseline trainers

- `train_a2c.py`: A2C, default hyperparameters, 1,000 steps.
- `train_ddpg.py`: DDPG, buffer 15,000, learning starts at 500, Gaussian action noise sigma 0.1, 1,000 steps.
- `train_sac.py`: SAC, buffer 15,000, learning starts at 500, 1,000 steps.
- `train_td3.py`: TD3, same replay settings and noise as DDPG, 1,000 steps.
- `train_trpo.py`: sb3-contrib TRPO, defaults, 1,000 steps.

The 1,000-step runs are far too small to claim convergence in a 210-dimensional continuous-control problem.

### `train_baselines.py` (73 lines)

- Trains DDPG, SAC, A2C, TD3, and TRPO sequentially for 1,000 steps each.
- Uses a replay buffer for off-policy algorithms and noise for DDPG/TD3.
- Comment incorrectly says `SatelliteEnv` lacks Lagrangian constraints.
- All baselines actually inherit the base environment’s adaptive safety penalty.

## 8. Evaluation, proof, and plotting code

### `test_env.py` (175 lines)

- Performs construction, reset, action, five-step, full-episode, and map-generation checks.
- Only asserts observation shape.
- Prints an expected observation range but does not assert it.
- Uses a random full episode and regenerates `graph_week4_env_test.png`.
- Runtime audit proved the range is invalid while script still printed “PASSED.”

### `step04_plot_training.py` (195 lines)

- Reads `models/latest_basic_run.txt` and matching `evaluations.npz`.
- If absent, silently creates a synthetic learning curve.
- Current pointer is absent, so rerunning creates a synthetic curve.
- Coverage map uses raw propagated positions; it does not apply a trained PPO policy despite its title.
- 22-degree circles are longitude/latitude plotting approximations, not physical footprints.

### `step05_safe_vs_unsafe.py` (281 lines)

- Evaluates Random, Basic PPO, and Safe PPO for 15 episodes with deterministic model actions.
- Uses `SafeSatelliteEnv` for all three agents.
- A missing model path becomes `None`; `None` means random actions.
- Since `latest_basic_run.txt` is absent, “Basic PPO” currently becomes a second random agent.
- Model load errors other than missing pointer are not handled.
- The summary metrics are based on the simplified environment.
- The two-panel world map plots the same raw satellite/debris coordinates on both sides; no policy trajectory is used. Titles imply different behaviors that are not present in plotted data.
- Caption says 10 satellites and 50 debris while config uses 70 and up to 500.

### `step06_compare_algorithms.py` (142 lines)

- Defines a real evaluation function but does not call it for the final output.
- Lines 74 onward explicitly generate “simulated theoretical evaluation metrics.”
- `all_results` hard-codes every reward, coverage, minimum distance, and collision result.
- `graph7_algorithm_comparison.png` is illustrative/fabricated and must not be cited as experimental evidence.

### `prove_agent_works.py` (480 lines)

- Loads latest Safe PPO when available.
- Selects one satellite and constructs a synthetic debris collision-course scenario.
- Compares no thrust, random actions, and Safe PPO.
- Builds a manually simulated state and saves 2D/3D outputs.
- Uses simplified custom trajectory equations, not the environment/Skyfield dynamics end-to-end.
- The synthetic case is useful as a demonstration harness, not mathematical proof.
- Final text saying it “proves” the formulation works is too strong.

### `step04_plot_training.py`, `visualize_data.py`, `skyfield_test.py`, `globe_3d_static.py`

- These are visualization scripts, not unit tests.
- They execute work at import time.
- Several titles say real-time/live even though data files are snapshots.
- Skyfield provides valid propagation from the stored elements, but stale elements reduce reliability at the current date.

## 9. Dashboard code

### `live_dashboard.py` (791 lines; primary dashboard)

Startup:

- Loads 70 Starlink objects and first 150 combined debris objects.
- Loads models through pointer text files.
- Missing models are omitted from the dropdown.
- Generates 15 synthetic debris objects with sinusoidally changing separation.

Position refresh:

- Skyfield propagates stored OMM objects to current time.
- Adds accumulated AI offsets to satellite Cartesian positions.
- On propagation failure, substitutes synthetic fallback orbits.
- Adds synthetic debris near selected satellites to guarantee visible threat-zone cycling.

Observation/model path:

- Reconstructs a 427-value observation compatible with saved model spaces.
- Coverage and fuel are set/overwritten in dashboard-specific ways.
- Calls selected model deterministically.
- Applies actions only to satellites currently in warning/danger zones. This is an extra hand-coded safety gate not present in training.
- Multiplies applied offsets by a display scale of 50, so visualization dynamics differ substantially from training.
- Does not apply the z action component to `_sat_thrust_acc` in the displayed transition.

Interpretation:

- It is a mixed real-data/synthetic-threat visualization.
- Displayed success cannot be attributed solely to the learned policy because of synthetic threats, action gating, action scaling, fallback states, and separate coverage logic.
- UI text calling it “Lagrangian CMDP PPO” should be softened.

### `live_dashboard2.py` (564 lines; older/alternate dashboard)

- Loads Safe and Basic PPO pointers if available.
- Advances a simulated time from a fixed start.
- Reconstructs a dashboard observation from lat/lon/alt and proximity features.
- Uses Safe model when present and includes simplified action effects.
- It is less complete than `live_dashboard.py` and should be treated as legacy unless intentionally revived.

### Diagnostics

- `diagnose_dashboard.py` duplicates primary dashboard position and synthetic-fragment calculations, prints nearest real pairs and expected dashboard zone.
- `verify_dashboard.py` simulates ten synthetic fragment distances across 15 ticks and checks whether multiple zones appear.
- These validate display mechanics, not learned collision avoidance.

### `run_ppo_demo.py` (59 lines)

- CLI orchestration script.
- Can force training, run proof only, dashboard only, or full flow.
- Calls scripts using the current Python executable.
- The dashboard subprocess blocks until stopped.

## 10. Data assets

All OMM JSON files share these keys:

`OBJECT_NAME`, `OBJECT_ID`, `EPOCH`, `MEAN_MOTION`, `ECCENTRICITY`, `INCLINATION`, `RA_OF_ASC_NODE`, `ARG_OF_PERICENTER`, `MEAN_ANOMALY`, `EPHEMERIS_TYPE`, `CLASSIFICATION_TYPE`, `NORAD_CAT_ID`, `ELEMENT_SET_NO`, `REV_AT_EPOCH`, `BSTAR`, `MEAN_MOTION_DOT`, `MEAN_MOTION_DDOT`.

Counts and timestamps:

| File | Records | Snapshot modified |
|---|---:|---|
| `active_satellites.json` | 15,007 | 2026-04-15 |
| `starlink_satellites.json` | 10,151 | 2026-04-15 |
| `iridium33_debris.json` | 108 | 2026-04-15 |
| `cosmos2251_debris.json` | 582 | 2026-04-15 |
| `fengyun1c_debris.json` | 1,867 | 2026-04-15 |
| `cosmos1408_debris.json` | 4 | 2026-04-15 |

The records are five months old at audit time. The downloader skips existing files larger than 1,000 bytes, so rerunning it will **not** refresh most datasets. `cosmos1408_debris.json` is only 2,216 bytes but also exceeds the skip threshold.

### `download_data.py` (150 lines)

- Downloads six CelesTrak groups only when the destination is absent/small.
- Writes 50 current Starlink subpoints to CSV.
- Computes active-satellite/debris distances and stores pairs below 8,000 km.
- Injects 100 synthetic high-risk rows, with 50 at 0.05-1.5 km and 50 at 2.0-9.9 km.
- Therefore `collision_training_data.csv` is a mixture of propagated and intentionally synthetic data.
- Despite its name, that CSV is not consumed by the RL environment or training code.

CSV schemas:

- `starlink_positions_now.csv`: 50 rows; `name,norad_id,latitude,longitude,altitude_km`.
- `collision_training_data.csv`: 7,897 rows; `sat_name,sat_x,sat_y,sat_z,deb_name,deb_x,deb_y,deb_z,distance_km,collision_risk`.
- Synthetic rows are identifiable by `(SYNTH)` in names.

`explore_data.py` prints descriptive statistics and distance-zone counts. `visualize_data.py` turns both CSVs into graphs 1-3.

## 11. Models and logs

### Pointer files

- `models/latest_safe_run.txt` points to `safe_ppo_20260720_230402/safe_final_model.zip`.
- Baseline pointers exist for A2C, DDPG, SAC, TD3, and TRPO.
- `models/latest_basic_run.txt` is missing.

### Saved model archives

All 12 current model ZIPs inspected have standard SB3 archive structure: `data`, Torch variable/policy files, optimizer state, SB3 version, and system information. DDPG/SAC/TD3 also contain actor/critic optimizer state; SAC contains entropy optimizer state.

Current model groups:

- A2C: one final model.
- DDPG: one final model.
- SAC: one final model.
- TD3: one final model.
- TRPO: one final model.
- Basic PPO: best model and 50,000-step checkpoint, but no final pointer.
- Safe PPO: three historical runs plus best, checkpoint, and final artifacts for the latest run.

### Evaluation NPZ files

`logs/basic_ppo_20260720_093815/evaluations.npz`:

- timesteps: 10,000 through 50,000, five checkpoints.
- results shape: 5 x 5 episodes; values approximately 483.99 to 528.99.
- episode lengths: 484 to 529.

`logs/safe_ppo_20260720_230402/evaluations.npz`:

- same checkpoint/episode shape.
- result range approximately 485.99 to 513.99.
- episode lengths: 486 to 514.

These rewards largely track episode length because each safe step yields about +1 and collision termination shortens episodes. They do not independently demonstrate coverage optimization or calibrated safety.

TensorBoard event files exist for several basic/safe runs but were not treated as authoritative experimental summaries because environment validity issues precede curve interpretation.

## 12. Critical defect register

Priority definitions: P0 invalidates central claims; P1 materially damages correctness; P2 affects maintainability/reproducibility; P3 cosmetic/documentation.

| ID | Priority | Defect | Impact |
|---|---|---|---|
| K1 | P0 | `step06_compare_algorithms.py` hard-codes results | Algorithm-comparison claims are not experimental |
| K2 | P0 | Basic and safe agents share Lagrangian penalty in base env | Safe-vs-unsafe experiment is not controlled |
| K3 | P0 | Coverage always 100% in verified episode | Reward is saturated; learning objective is trivial |
| K4 | P0 | Distance risk is called collision probability | Safety metric is not calibrated physical Pc |
| K5 | P0 | Simplified dynamics/action units are inconsistent | Maneuvers are not physically defensible |
| K6 | P1 | Observation violates declared Box bounds | Official SB3 environment checker fails |
| K7 | P1 | Lambda is not reset | Episodes are not independent/Markov-consistent |
| K8 | P1 | Missing basic pointer silently becomes random | Comparison labels can be false |
| K9 | P1 | Comparison world maps use identical positions | Graphic implies behavior not present in data |
| K10 | P1 | Dashboard uses hand gating and 50x action scale | Dashboard behavior differs from trained environment |
| K11 | P1 | Baselines trained only 1,000 steps | No convergence/fairness basis |
| K12 | P1 | Stale TLE/OMM records propagated to current time | Large errors and invalid normalized states possible |
| K13 | P2 | Downloader skips existing snapshots | “Refresh” command does not refresh |
| K14 | P2 | No dependency/lock file | Environment is not reproducible |
| K15 | P2 | Hard-coded absolute paths | Project runs only in the original location |
| K16 | P2 | Bare `except` blocks | Data/propagation defects are hidden |
| K17 | P2 | Scripts execute at import time | Hard to test and reuse safely |
| K18 | P2 | Generated, source, archive, and cache files mixed | Workspace is difficult to maintain |
| K19 | P3 | Documentation filenames/settings are inconsistent | Confuses presentation and usage |

## 13. Recommended repair sequence

1. Create separate `BaseSatelliteEnv` dynamics, `UnsafeRewardWrapper`, and `LagrangianSafetyWrapper`.
2. Reset every episode variable, especially lambda.
3. Normalize/clip observations consistently and make `check_env()` pass.
4. Replace action displacement with a dimensionally consistent delta-v/velocity transition.
5. Replace the coverage proxy with altitude-aware horizon/elevation geometry.
6. Replace “collision probability” terminology or implement covariance-based Pc.
7. Construct deterministic conjunction scenarios so safety events are learnable and repeatable.
8. Make missing models fatal in evaluations.
9. Evaluate every actual saved model; delete hard-coded result tables.
10. Use identical train/eval scenarios, budgets, seeds, and metrics across algorithms.
11. Add unit tests for reset, bounds, dynamics, reward terms, termination, pointer loading, and evaluation integrity.
12. Add `pyproject.toml`/lockfile, relative paths, logging, Git, and artifact separation.
13. Regenerate plots, dashboard labels, PDFs, and slides only after corrected experiments.

## 14. Output artifact catalog and trust status

### Static PNG plots

All current top-level plot PNGs opened as valid images with these dimensions:

- `graph1_altitude_distribution.png`: 1484 x 881.
- `graph2_satellite_world_map.png`: 2181 x 1128.
- `graph3_collision_risk.png`: 2082 x 923.
- `graph_step03_live_positions.png`: 2373 x 1293.
- `graph_step03_all5_future_tracks.png`: 2373 x 1293.
- `graph_week4_env_test.png`: 2381 x 1309.
- `graph4_learning_curve.png`: 1634 x 881.
- `graph5_coverage_on_map.png`: 2683 x 1023.
- `graph6_safe_vs_unsafe.png`: 2085 x 1478.
- `graph6b_worldmap_safe_vs_unsafe.png`: 2683 x 1023.
- `graph7_algorithm_comparison.png`: 1800 x 900.
- `graph10_proof_2d_comparison.png`: 2384 x 1475.

Trust classifications:

- Graphs 1-3: descriptive snapshots; collision CSV includes injected synthetic rows.
- Step03/Skyfield maps: valid visualizations of stored orbital records propagated at run time, subject to stale elements.
- Graph 4: may be synthetic because the basic pointer is absent.
- Graph 5: raw satellite positions/footprints, not learned-policy output.
- Graphs 6/6b: evaluation is compromised; world-map sides do not show policy-specific trajectories.
- Graph 7: explicitly hard-coded, illustrative only.
- Graph 10: synthetic scenario demonstration, not proof.

### Interactive HTML

- `plots/3d_globe_satellite_map.html`: Plotly globe from stored orbital data.
- `plots/graph10_proof_3d_earth.html`: synthetic proof-scenario visualization.
- Matching copies under `_hidden_completed_work` are historical archives.

### Reports and presentations

- `Algorithm_Explanation_Report.pdf`: 13-page older generated report. Contains overstated guarantees, mismatched configuration values, inaccurate architecture descriptions, and hard-coded result claims.
- `output/pdf/Satellite_Safe_PPO_Presentation_Report.pdf`: 13-page corrected presentation guide generated during this session; visually rendered and checked.
- `_hidden_completed_work/presentation/Minor Project.docx`: historical project report.
- Historical PDFs/PPTX files: satellite project and constellation coverage presentations.
- `generate_pdf.py`: 706-line generator for the older algorithm report; its content preserves the old overclaims and should not be used without revision.
- `build_presentation_report.py`: 249-line generator for the corrected report.

### Research papers

The `reserch pepar` directory contains 12 reference PDFs covering spacecraft proximity maneuvering, PPO/TRPO-era RL, constrained/safe RL, scheduling, coverage optimization, and recent satellite-related RL work. These are literature inputs, not executable project components. Filenames 1-12 are the stable identifiers. Verify individual paper metadata and claims directly before citing them in academic writing.

### Historical archive

`_hidden_completed_work` contains earlier source versions, logs, one Safe PPO model, plots, and presentation files. Treat this directory as an archive, not the active implementation. Do not edit it unless explicitly restoring or comparing history.

### Temporary files

`tmp/pdfs` contains rendered PNGs used to inspect source/final PDFs. These are QA intermediates, not deliverables.

## 15. Dependencies

Observed runtime dependencies:

- Python 3.x
- numpy, pandas
- torch
- gymnasium
- stable-baselines3
- sb3-contrib
- skyfield and its SGP4 dependencies
- requests
- matplotlib
- geopandas and geodatasets
- plotly
- dash
- reportlab
- pypdf/pdfplumber for artifact inspection

Potentially loaded indirectly in the current Python environment: TensorFlow/TensorBoard components emit oneDNN messages during SB3 import. This is not part of the project’s declared dependency setup.

No `requirements.txt`, `pyproject.toml`, environment file, or lockfile exists.

## 16. File-by-file quick index

| File | Role | Read before changing |
|---|---|---|
| `config.py` | Global paths/hyperparameters | Any environment/training work |
| `download_data.py` | Download and CSV preprocessing | Data refresh/ETL |
| `explore_data.py` | Console data statistics | Data analysis |
| `visualize_data.py` | Graphs 1-3 | Dataset figures |
| `skyfield_test.py` | Five-satellite propagation/maps | Skyfield demonstrations |
| `globe_3d_static.py` | Interactive 3D globe | Static orbital visualization |
| `env.py` | Main environment/dynamics/reward | Core RL work |
| `safe_env.py` | Cost logging wrapper | Safety metrics |
| `test_env.py` | Smoke/performance/map test | Environment verification |
| `train_basic.py` | Basic PPO trainer | PPO baseline |
| `train_safe.py` | PPO plus safety logging | Main Safe PPO training |
| `train_a2c.py` | A2C 1k trainer | Baseline |
| `train_ddpg.py` | DDPG 1k trainer | Baseline |
| `train_sac.py` | SAC 1k trainer | Baseline |
| `train_td3.py` | TD3 1k trainer | Baseline |
| `train_trpo.py` | TRPO 1k trainer | Baseline |
| `train_baselines.py` | Batch baseline orchestrator | Comparative training |
| `step04_plot_training.py` | Learning curve/coverage map | Training figures |
| `step05_safe_vs_unsafe.py` | Three-agent comparison | PPO comparison |
| `step06_compare_algorithms.py` | Hard-coded seven-agent chart | Must repair before use |
| `prove_agent_works.py` | Synthetic three-scenario demo | Proof/demo figures |
| `live_dashboard.py` | Primary multi-algorithm dashboard | Live demo |
| `live_dashboard2.py` | Legacy alternate dashboard | Only if explicitly requested |
| `diagnose_dashboard.py` | Primary dashboard math diagnostic | Dashboard troubleshooting |
| `verify_dashboard.py` | Synthetic zone-cycle diagnostic | Dashboard threat cycling |
| `run_ppo_demo.py` | Train/proof/dashboard launcher | Full demo orchestration |
| `generate_pdf.py` | Old inaccurate PDF generator | Avoid for new reporting |
| `build_presentation_report.py` | Corrected presentation PDF generator | Current presentation report |
| `RUN_GUIDE.md` | Week-by-week commands | Usage overview; contains stale claims |

## 17. Safe command guide

Read-only/smoke checks:

```powershell
python -m compileall -q .
python -c "from env import SatelliteEnv; e=SatelliteEnv(); print(e.reset(seed=42)[0].shape)"
python diagnose_dashboard.py
python verify_dashboard.py
```

Full environment test (regenerates a plot):

```powershell
python test_env.py
```

Training (expensive and writes models/logs):

```powershell
python train_basic.py
python train_safe.py
python train_baselines.py
```

Dashboard:

```powershell
python live_dashboard.py
# open http://localhost:8050
```

Do not run `step06_compare_algorithms.py` to create evidence until its hard-coded results are removed.

## 18. Future-agent operating rules

1. Start with this knowledge base.
2. Check modification times or diffs before trusting it; update this file after material code/data changes.
3. Open only the specific files named in Section 16 for the requested subsystem.
4. Never cite graph 7 or old report metrics as experimental evidence.
5. Never silently substitute a random policy for a missing model.
6. Preserve the distinction between real orbital-record initialization, simplified training dynamics, and synthetic dashboard/proof threats.
7. Use “risk proxy” unless physical probability-of-collision calculations are implemented.
8. Use “adaptive safety penalty” unless a genuine constrained optimizer is implemented.
9. Run `check_env()` after every environment change.
10. Re-run controlled model evaluations and regenerate all dependent artifacts after reward/dynamics/state changes; old models become incompatible in meaning even if their tensor shapes still load.

## 19. Audit scope and limits

Every active top-level Python source was mapped by imports, definitions, control flow, file I/O, and cross-file dependencies. Core environment, training, comparison, dashboard, data, reporting, and verification paths received detailed line-level review. Datasets were inspected for schema and record counts. Model archives were inspected for SB3 structure, and the latest Safe PPO model was loaded. NPZ logs were inspected numerically. PNG outputs were checked for validity and dimensions. The current and previous presentation PDFs were rendered/inspected in the earlier report task.

Binary PDFs/PPTX/DOCX/model weights and research papers were not interpreted byte-by-byte; that would not provide meaningful code analysis. They were classified by metadata, container structure, role, and relationship to generating code. Individual research papers should be opened only when their scientific content is needed.

This document records the state of the workspace at the audit date; it is not automatically synchronized after edits.
