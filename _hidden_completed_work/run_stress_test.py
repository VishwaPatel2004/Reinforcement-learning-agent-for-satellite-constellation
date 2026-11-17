# =============================================================================
# run_stress_test.py
# COLLISION STRESS TEST — Prove the Safe RL Agent Actually Works
#
# WHAT THIS DOES:
#   Runs 3 agents inside the StressTestEnv (all real debris + 5 synthetic
#   debris on guaranteed collision courses) and compares:
#
#   Agent 1: RANDOM     — takes random actions (baseline, expected: crashes)
#   Agent 2: BASIC PPO  — if model saved, else random (expected: may crash)
#   Agent 3: SAFE PPO   — our trained Safe RL agent (expected: avoids all)
#
# KEY METRIC: Did each agent SURVIVE 20 episodes without a collision?
# =============================================================================

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from stable_baselines3 import PPO
from stress_test_env import StressTestEnv
import config as cfg

print("\n" + "=" * 70)
print("COLLISION STRESS TEST — PROOF THAT SAFE RL AGENT WORKS")
print("=" * 70)
print()
print("HOW IT WORKS:")
print("  → ALL debris types loaded (Iridium-33 + Cosmos-2251 + Fengyun-1C + Cosmos-1408)")
print("  → 5 synthetic debris pieces injected ON COLLISION COURSE each episode")
print("  → Each synthetic debris starts 150 km away and closes at 0.5 km/step")
print("  → Without thrust: guaranteed collision in ~300 steps")
print("  → Safe PPO must detect the threat and thrust away to survive")
print()

# ── Load Models ───────────────────────────────────────────────────────────────
try:
    with open(os.path.join(cfg.MODELS_DIR, "latest_basic_run.txt")) as f:
        basic_path = f.read().strip().split('\n')[1]
    basic_model = PPO.load(basic_path)
    print(f"✅ Basic PPO loaded: {basic_path}")
except Exception as e:
    basic_model = None
    print(f"⚠️  Basic PPO not found ({e}) — will use random agent as substitute")

try:
    with open(os.path.join(cfg.MODELS_DIR, "latest_safe_run.txt")) as f:
        safe_path = f.read().strip().split('\n')[1]
    safe_model = PPO.load(safe_path)
    print(f"✅ Safe PPO  loaded: {safe_path}")
except Exception as e:
    safe_model = None
    print(f"⚠️  Safe PPO not found ({e})")

print()

# ── Evaluation Function ────────────────────────────────────────────────────────
def run_stress_episodes(model, label, n_episodes=20):
    env = StressTestEnv(render_mode="human", n_synthetic=5)
    results = []

    print(f"\n{'─'*60}")
    print(f"TESTING: {label} ({n_episodes} episodes with forced collisions)")
    print(f"{'─'*60}")

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep * 31)
        ep_reward   = 0
        collided    = False
        min_dist_ep = float('inf')
        thrusts_ep  = []
        step_count  = 0

        while True:
            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward   += reward
            step_count  += 1
            thrusts_ep.append(float(np.sum(np.abs(action))))

            if info['min_distance_km'] < min_dist_ep:
                min_dist_ep = info['min_distance_km']

            if step_count % 60 == 0:   # print every 60 steps
                env.render()

            if terminated:
                collided = True
                print(f"  💥 Episode {ep+1:2d}: COLLISION at step {step_count} "
                      f"| Min dist: {min_dist_ep:.1f} km")
                break
            if truncated:
                print(f"  ✅ Episode {ep+1:2d}: SURVIVED 540 steps "
                      f"| Min dist: {min_dist_ep:.1f} km "
                      f"| Avg thrust: {np.mean(thrusts_ep):.5f}")
                break

        results.append({
            "ep":          ep + 1,
            "collided":    collided,
            "min_dist":    min_dist_ep,
            "reward":      ep_reward,
            "avg_thrust":  np.mean(thrusts_ep),
            "steps":       step_count,
        })

    # Summary
    n_survived  = sum(1 for r in results if not r["collided"])
    avg_min_d   = np.mean([r["min_dist"]   for r in results])
    avg_thrust  = np.mean([r["avg_thrust"] for r in results])
    avg_reward  = np.mean([r["reward"]     for r in results])

    print(f"\n  ── {label} SUMMARY ──")
    print(f"  Survived   : {n_survived}/{n_episodes} episodes")
    print(f"  Collision % : {(1 - n_survived/n_episodes)*100:.0f}%")
    print(f"  Avg min dist: {avg_min_d:.1f} km")
    print(f"  Avg thrust  : {avg_thrust:.5f}")
    print(f"  Avg reward  : {avg_reward:.2f}")

    return results, {
        "label":        label,
        "n_survived":   n_survived,
        "n_episodes":   n_episodes,
        "avg_min_dist": avg_min_d,
        "avg_thrust":   avg_thrust,
        "avg_reward":   avg_reward,
    }

# ── Run all 3 agents ──────────────────────────────────────────────────────────
N_EPISODES = 20

rand_results,  rand_summary  = run_stress_episodes(None,        "Random Agent", N_EPISODES)
basic_results, basic_summary = run_stress_episodes(basic_model, "Basic PPO",    N_EPISODES)
safe_results,  safe_summary  = run_stress_episodes(safe_model,  "Safe PPO (CMDP)", N_EPISODES)

summaries = [rand_summary, basic_summary, safe_summary]

# ── Print Final Table ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"{'FINAL STRESS TEST RESULTS':^70}")
print("=" * 70)
print(f"{'Metric':<30} {'Random':>12} {'Basic PPO':>12} {'Safe PPO':>12}")
print("-" * 70)
print(f"{'Survived (out of 20)':<30} "
      f"{rand_summary['n_survived']:>12} "
      f"{basic_summary['n_survived']:>12} "
      f"{safe_summary['n_survived']:>12}")
print(f"{'Collision Rate':<30} "
      f"{(1 - rand_summary['n_survived']/N_EPISODES)*100:>11.0f}% "
      f"{(1 - basic_summary['n_survived']/N_EPISODES)*100:>11.0f}% "
      f"{(1 - safe_summary['n_survived']/N_EPISODES)*100:>11.0f}%")
print(f"{'Avg Min Distance (km)':<30} "
      f"{rand_summary['avg_min_dist']:>12.1f} "
      f"{basic_summary['avg_min_dist']:>12.1f} "
      f"{safe_summary['avg_min_dist']:>12.1f}")
print(f"{'Avg Thrust Applied':<30} "
      f"{rand_summary['avg_thrust']:>12.5f} "
      f"{basic_summary['avg_thrust']:>12.5f} "
      f"{safe_summary['avg_thrust']:>12.5f}")
print(f"{'Avg Episode Reward':<30} "
      f"{rand_summary['avg_reward']:>12.2f} "
      f"{basic_summary['avg_reward']:>12.2f} "
      f"{safe_summary['avg_reward']:>12.2f}")
print("=" * 70)

# ── Generate Proof Graph ───────────────────────────────────────────────────────
print("\nGenerating stress test proof graph...")

fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.patch.set_facecolor('#0d1b2a')
fig.suptitle(
    "STRESS TEST RESULTS: Safe RL vs Unsafe Agents\n"
    "Synthetic debris injected on guaranteed collision course — all debris types loaded\n"
    "Iridium-33 + Cosmos-2251 + Fengyun-1C + Cosmos-1408",
    color='white', fontsize=13, fontweight='bold'
)

colors  = ['#e74c3c', '#f39c12', '#2ecc71']
labels  = ['Random\nAgent', 'Basic PPO\n(No Safety)', 'Safe PPO\n(CMDP — Ours)']

# Panel 1: Survival rate
ax = axes[0]
ax.set_facecolor('#1a2a3a')
survived = [s['n_survived'] for s in summaries]
bars = ax.bar(range(3), survived, color=colors, edgecolor='white', linewidth=1.2, width=0.6)
ax.set_xticks(range(3))
ax.set_xticklabels(labels, color='white', fontsize=10)
ax.set_ylabel("Episodes Survived (out of 20)", color='white', fontsize=11)
ax.set_title("Survival Rate", color='white', fontsize=13, fontweight='bold')
ax.set_ylim(0, 22)
ax.tick_params(colors='white')
ax.grid(True, alpha=0.2, axis='y')
for spine in ax.spines.values(): spine.set_edgecolor('#334455')
for bar, val in zip(bars, survived):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val}/20', ha='center', va='bottom', color='white',
            fontweight='bold', fontsize=13)
ax.axhline(20, color='white', linestyle='--', alpha=0.3, linewidth=1)

# Panel 2: Average minimum distance
ax = axes[1]
ax.set_facecolor('#1a2a3a')
min_dists = [s['avg_min_dist'] for s in summaries]
bars = ax.bar(range(3), min_dists, color=colors, edgecolor='white', linewidth=1.2, width=0.6)
ax.set_xticks(range(3))
ax.set_xticklabels(labels, color='white', fontsize=10)
ax.set_ylabel("Avg Minimum Distance to Debris (km)", color='white', fontsize=11)
ax.set_title("Debris Avoidance Distance", color='white', fontsize=13, fontweight='bold')
ax.tick_params(colors='white')
ax.grid(True, alpha=0.2, axis='y')
for spine in ax.spines.values(): spine.set_edgecolor('#334455')
ax.axhline(cfg.SAFE_DISTANCE_KM, color='#e74c3c', linestyle='--', linewidth=2,
           label=f'Safe distance ({cfg.SAFE_DISTANCE_KM} km)')
ax.legend(fontsize=9, facecolor='#0d1b2a', edgecolor='white', labelcolor='white')
for bar, val in zip(bars, min_dists):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.0f} km', ha='center', va='bottom', color='white',
            fontweight='bold', fontsize=12)

# Panel 3: Min distance per episode for Safe PPO only (proof line graph)
ax = axes[2]
ax.set_facecolor('#1a2a3a')
safe_min_dists  = [r['min_dist'] for r in safe_results]
rand_min_dists  = [r['min_dist'] for r in rand_results]
basic_min_dists = [r['min_dist'] for r in basic_results]
eps = list(range(1, N_EPISODES + 1))

ax.plot(eps, rand_min_dists,  color='#e74c3c', lw=2, marker='o', ms=5, label='Random')
ax.plot(eps, basic_min_dists, color='#f39c12', lw=2, marker='s', ms=5, label='Basic PPO')
ax.plot(eps, safe_min_dists,  color='#2ecc71', lw=2, marker='*', ms=8, label='Safe PPO (CMDP)')
ax.axhline(cfg.SAFE_DISTANCE_KM, color='white', linestyle='--', linewidth=1.5,
           alpha=0.6, label=f'Safe distance ({cfg.SAFE_DISTANCE_KM} km)')
ax.axhline(1.0, color='#e74c3c', linestyle=':', linewidth=1.5,
           alpha=0.8, label='Collision threshold (1 km)')
ax.set_xlabel("Episode", color='white', fontsize=11)
ax.set_ylabel("Min Distance to Debris (km)", color='white', fontsize=11)
ax.set_title("Per-Episode: Min Distance to Debris\n(Higher = Safer)", color='white',
             fontsize=12, fontweight='bold')
ax.tick_params(colors='white')
ax.grid(True, alpha=0.2)
for spine in ax.spines.values(): spine.set_edgecolor('#334455')
ax.legend(fontsize=9, facecolor='#0d1b2a', edgecolor='white', labelcolor='white')

plt.tight_layout()
out_path = os.path.join(cfg.PLOTS_DIR, "graph9_stress_test_proof.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d1b2a')
plt.close()
print(f"✅ Saved: {out_path}")

print("\n" + "=" * 70)
print("STRESS TEST COMPLETE")
print()
safe_survived = safe_summary['n_survived']
if safe_survived >= 18:
    print(f"  ✅ RESULT: Safe PPO PROVES IT WORKS!")
    print(f"     Survived {safe_survived}/20 episodes with guaranteed collision courses.")
    print(f"     The agent learned to thrust away from approaching debris.")
elif safe_survived >= 12:
    print(f"  🟡 RESULT: Safe PPO partially works ({safe_survived}/20 survived).")
    print(f"     Needs more training steps to fully learn avoidance.")
else:
    print(f"  ⚠️  RESULT: Safe PPO needs more training ({safe_survived}/20 survived).")
    print(f"     Increase TOTAL_TIMESTEPS in config.py and retrain.")
print()
print(f"  See graph: {out_path}")
print("=" * 70)
