# =============================================================================
# train_basic.py
# WEEK 5 DELIVERABLE: First PPO Training Run
#
# This script trains a basic PPO agent (WITHOUT safety constraint).
# The safety constraint will be added in Week 6.
# This is the BASELINE model.
#
# Expected training time: 45-90 minutes on CPU laptop
# =============================================================================

import os
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from env import SatelliteEnv
import config as cfg

# ── Setup ─────────────────────────────────────────────────────────────────────
run_id   = f"basic_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
save_dir = os.path.join(cfg.MODELS_DIR, run_id)
log_dir  = os.path.join(cfg.LOGS_DIR, run_id)
os.makedirs(save_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

print("=" * 60)
print("WEEK 5: BASIC PPO TRAINING (No Safety Constraint)")
print("=" * 60)
print(f"Run ID     : {run_id}")
print(f"Save dir   : {save_dir}")
print(f"Total steps: {cfg.TOTAL_TIMESTEPS:,}")
print()

# ── Create Environments ────────────────────────────────────────────────────────
print("Creating environments...")
train_env = DummyVecEnv([lambda: Monitor(SatelliteEnv())])
eval_env  = DummyVecEnv([lambda: Monitor(SatelliteEnv())])
print("✅ Environments ready\n")

# ── Create Callbacks ───────────────────────────────────────────────────────────
eval_cb = EvalCallback(
    eval_env,
    best_model_save_path = save_dir,
    log_path             = log_dir,
    eval_freq            = 10_000,
    n_eval_episodes      = 5,
    deterministic        = True,
    verbose              = 1
)

ckpt_cb = CheckpointCallback(
    save_freq   = 50_000,
    save_path   = save_dir,
    name_prefix = "ckpt",
    verbose     = 1
)

# ── Create PPO Agent ───────────────────────────────────────────────────────────
print("Creating PPO agent...")
model = PPO(
    policy          = "MlpPolicy",
    env             = train_env,
    learning_rate   = cfg.LEARNING_RATE,
    n_steps         = cfg.N_STEPS,
    batch_size      = cfg.BATCH_SIZE,
    n_epochs        = cfg.N_EPOCHS,
    gamma           = cfg.GAMMA,
    gae_lambda      = 0.95,
    clip_range      = 0.2,
    ent_coef        = 0.01,
    verbose         = 1,
    tensorboard_log = log_dir,
    seed            = cfg.SEED
)

total_params = sum(p.numel() for p in model.policy.parameters())
print(f"✅ PPO agent created")
print(f"   Neural network parameters: {total_params:,}")
print()

# ── TRAIN ──────────────────────────────────────────────────────────────────────
print("STARTING TRAINING...")
print("Watch 'ep_rew_mean' — it should INCREASE over time.")
print("Press Ctrl+C to stop early if needed.\n")
print("-" * 60)

model.learn(
    total_timesteps    = cfg.TOTAL_TIMESTEPS,
    callback           = [eval_cb, ckpt_cb],
    progress_bar       = True,
    reset_num_timesteps = True
)

# ── Save Final Model ───────────────────────────────────────────────────────────
final_path = os.path.join(save_dir, "final_model")
model.save(final_path)
print(f"\n✅ Final model saved: {final_path}.zip")

# ── Save Run ID for reference ──────────────────────────────────────────────────
with open(os.path.join(cfg.MODELS_DIR, "latest_basic_run.txt"), 'w') as f:
    f.write(run_id)
    f.write(f"\n{final_path}.zip")
print(f"✅ Run ID saved for reference")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("Next step: Run step04_plot_training.py to see results")
print("=" * 60)