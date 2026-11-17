import os
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback
)

from safe_env import SafeSatelliteEnv
import config as cfg

class SafetyLogCallback(BaseCallback):
    """Logs safety-specific metrics during training."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.ep_costs    = []
        self.ep_rewards  = []

    def _on_step(self):
        for info in self.locals.get('infos', []):
            if 'avg_ep_cost' in info:
                self.ep_costs.append(info['avg_ep_cost'])
            if 'cost' in info:
                
                self.logger.record("safety/collision_prob", info['cost'])
                self.logger.record("safety/avg_ep_cost",    info.get('avg_ep_cost', 0))
                self.logger.record("safety/lambda",          info.get('lambda_cost', 0))
                self.logger.record("safety/constraint_ok",
                                   float(info.get('constraint_ok', False)))
        return True

run_id   = f"safe_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
save_dir = os.path.join(cfg.MODELS_DIR, run_id)
log_dir  = os.path.join(cfg.LOGS_DIR, run_id)
os.makedirs(save_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

print("=" * 60)
print("WEEK 6: SAFE PPO TRAINING (With Lagrangian Constraint)")
print("=" * 60)
print(f"Cost limit  : {cfg.COST_LIMIT} (collision prob must stay below this)")
print(f"Lambda init : {cfg.LAMBDA_INIT}")
print(f"Lambda lr   : {cfg.LAMBDA_LR}")
print(f"Lambda max  : {cfg.LAMBDA_MAX}")
print()


train_env = DummyVecEnv([lambda: Monitor(SafeSatelliteEnv())])
eval_env  = DummyVecEnv([lambda: Monitor(SafeSatelliteEnv())])


model = PPO(
    policy          = "MlpPolicy",
    env             = train_env,
    learning_rate   = cfg.LEARNING_RATE,
    n_steps         = cfg.N_STEPS,
    batch_size      = cfg.BATCH_SIZE,
    n_epochs        = cfg.N_EPOCHS,
    gamma           = cfg.GAMMA,
    verbose         = 1,
    tensorboard_log = log_dir,
    seed            = cfg.SEED
)


callbacks = [
    EvalCallback(eval_env,
                 best_model_save_path=save_dir,
                 log_path=log_dir,
                 eval_freq=10_000,
                 n_eval_episodes=5,
                 verbose=1),
    CheckpointCallback(save_freq=50_000, save_path=save_dir, verbose=1),
    SafetyLogCallback(verbose=0)
]


print("STARTING SAFE RL TRAINING...")
print("Watch 'ep_rew_mean' increase and collision prob stay low.\n")


try:
    model.learn(
        total_timesteps    = cfg.TOTAL_TIMESTEPS,
        callback           = callbacks,
        progress_bar       = True,
        reset_num_timesteps = True
    )
except KeyboardInterrupt:
    print("\nTraining stopped early! Saving model anyway for demo purposes...")

final_path = os.path.join(save_dir, "safe_final_model")
model.save(final_path)

# Save run ID
with open(os.path.join(cfg.MODELS_DIR, "latest_safe_run.txt"), 'w') as f:
    f.write(run_id + "\n" + final_path + ".zip")

print("\n" + "=" * 60)
print("SAFE TRAINING COMPLETE!")
print("Next: Run step05_safe_vs_unsafe.py to see comparison")
print("=" * 60)
