# 🚀 Satellite RL Project: Master State & Roadmap
**Project Name:** Safe Deep Reinforcement Learning for Satellite Constellation Coverage and Autonomous Collision Avoidance
**Objective:** A 6-month research-grade minor/major project aimed at replacing manual satellite trajectory calculations with an autonomous AI system.

*This document serves as the "Master Memory" to quickly resume the project in any future session.*

---

## 1. 🎯 Core Architecture We Selected
We are formulating the problem as a **Constrained Markov Decision Process (CMDP)** to solve a Multi-Objective problem using **Safe Reinforcement Learning**.

*   **Objective 1 (Reward - $R_t$):** Maximize internet/communication coverage footprint.
*   **Objective 2 (Constraint - $C_t$):** Ensure collision risk remains below a strict safety threshold ($d$) using Lagrangian Multipliers.
*   **The AI Brain:** Multi-Layer Perceptron (MLP) mapping 3D spatial data to continuous/discrete thruster actions.
*   **The RL Algorithms:** We start with **DQN** (Baseline/Proof of Concept), upgrade to **PPO** (Core Algorithm), and finally extend to **MAPPO** (Multi-Agent for swarm coordination).

---

## 2. 🗄️ Dataset Approach We Locked In
Instead of relying purely on mathematically generated synthetic keep-out zones (like older papers), we successfully bridged synthetic physics with real-world hazards.

*   **Data Source Selected:** CelesTrak.org / Space-Track
*   **Coverage Data:** We downloaded actual TLEs for the **Starlink** mega-constellation.
*   **Collision Data:** We downloaded actual TLEs for the **Iridium-33** and **Cosmos-1408** debris clouds.
*   **Current State:** We wrote a Python script (`download_data.py`) using the `skyfield` physics engine. It successfully converted raw TLE strings into `starlink_positions_now.csv` (lat/lon for coverage) and `collision_training_data.csv` (pairwise XYZ distances providing the exact $C_t$ collision risk constraint).
*   **Why this is genius:** It uses real-world starting points, but `skyfield` continuously propagates their motion forward in time, creating incredibly realistic, continuously-moving scenarios for the agent to navigate.

---

## 3. 📚 The Literature Foundation We Established
We curated a highly targeted list of 10 research papers. The core narrative we built for Professor Jindal revolves around these anchor papers:

1.  **Lü et al., IEEE 2022 ("Spacecraft Proximity Maneuvering... DQN"):** This is our **baseline**. We are extending their single-spacecraft, synthetic-obstacle DQN approach by scaling it to a constellation, adding a coverage objective, and using real CelesTrak hazard data.
2.  **Patnala & Abdin, Advances in Space Res. 2025 ("On-orbit servicing framework... RL"):** This justifies our use of RL for maneuvering and confirms that predicting collision probabilities from orbital data is the correct state representation.
3.  **Hassan et al., IEEE 2023 ("SpaceRIS LEO Sat Coverage... MAPPO"):** This justifies our final expansion phase for coverage maximization using Multi-Agent Proximal Policy Optimization.
4.  **Achiam et al., OpenAI 2017 ("Constrained Policy Optimization"):** The theoretical mathematical backbone for how we enforce the zero-collision Lagrangian rule.

---

## 4. ✅ What We Have Accomplished So Far
1.  **Project Roadmap:** Defined the 6-month roadmap, proving we can accelerate development into a 2-month build cycle since data collection is finished.
2.  **Presentation Generation:** Wrote a `python-pptx` script to generate a 14-slide, dark-themed, highly professional PowerPoint (`.pptx` and `.html`) for presenting the project idea to the professor.
3.  **Dataset Acquisition:** Successfully wrote and executed `download_data.py`, retrieving the OMM JSON datasets, processing them with `skyfield.EarthSatellite.from_omm`, and saving out the workable `.csv` state formats.
4.  **Academic Positioning:** Scripted the exact arguments to use when distinguishing this project from existing simulated-only IEEE papers.

---

## 5. 🔜 The Future Workflow (Next Steps)
When resuming this project in the next session, jump directly into these sequential milestones:

### Phase 1: Environment Construction (`env.py`)
*   **Task:** Build the custom OpenAI `gymnasium` environment.
*   **Process:** Load the `csv` data into the environment's `__init__`. Code the `step()` function to update satellite locations, calculate the coverage calculation (Reward), and check the distance between satellites and debris (Constraint).

### Phase 2: Single-Agent Training Pipeline (`train_ppo.py`)
*   **Task:** Set up the RL training loop.
*   **Process:** Use a library like `Stable-Baselines3`. Train the PPO agent on the `gymnasium` environment. Tune hyperparameters (learning rate, discount factor, Lagrangian multiplier) until the agent consistently achieves high coverage without colliding.

### Phase 3: Evaluation and Visualization (`render.py`)
*   **Task:** Prove that the agent is working.
*   **Process:** Code a visualizer (using Matplotlib or PyGame) to show a 2D/3D map of Earth. Watch the agent dynamically maneuver the active satellites away from incoming red debris markers while keeping maximum spread.

### Phase 4: Multi-Agent Extension (Optional Finale)
*   **Task:** Upgrade the system to MAPPO (Multi-Agent PPO) using `Ray RLlib` or `PettingZoo`.
*   **Process:** Allow each satellite to act as an individual brain to demonstrate decentralized swarm intelligence.
