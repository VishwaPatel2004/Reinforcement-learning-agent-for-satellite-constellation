import os
import numpy as np
from datetime import datetime
from stable_baselines3 import DDPG, SAC, A2C, TD3
from sb3_contrib import TRPO
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from env import SatelliteEnv
import config as cfg

def train_algorithm(algo_name, algo_class, total_timesteps=cfg.TOTAL_TIMESTEPS):
    print("=" * 60)
    print(f"TRAINING BASELINE: {algo_name}")
    print("=" * 60)
    
    run_id = f"{algo_name.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir = os.path.join(cfg.MODELS_DIR, "baselines", run_id)
    os.makedirs(save_dir, exist_ok=True)
    
    # We use the standard SatelliteEnv (which lacks Lagrangian constraints but has the -1000 penalty)
    env = DummyVecEnv([lambda: Monitor(SatelliteEnv())])
    
    kwargs = {
        "policy": "MlpPolicy",
        "env": env,
        "verbose": 0,
        "seed": cfg.SEED
    }
    
    # Off-policy deterministic algorithms need explicit action noise to explore
    if algo_name in ["DDPG", "TD3", "SAC"]:
        # Limit buffer size to prevent RAM overflow on Windows
        kwargs["buffer_size"] = 15000
        kwargs["learning_starts"] = 500
        if algo_name in ["DDPG", "TD3"]:
            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
            kwargs["action_noise"] = action_noise
    
    # Initialize model
    model = algo_class(**kwargs)
    
    print(f"Starting training for {algo_name} ({total_timesteps} steps)...")
    try:
        model.learn(total_timesteps=total_timesteps, progress_bar=True)
    except KeyboardInterrupt:
        print(f"\nTraining for {algo_name} stopped early by user.")
        
    final_path = os.path.join(save_dir, f"{algo_name.lower()}_final_model")
    model.save(final_path)
    
    # Save a reference txt for easy loading later
    with open(os.path.join(cfg.MODELS_DIR, "baselines", f"latest_{algo_name.lower()}.txt"), 'w') as f:
        f.write(run_id + "\n" + final_path + ".zip")
        
    print(f"{algo_name} training complete and saved to {final_path}.zip\n")

if __name__ == "__main__":
    os.makedirs(os.path.join(cfg.MODELS_DIR, "baselines"), exist_ok=True)
    
    algorithms = {
        "DDPG": DDPG,
        "SAC": SAC,
        "A2C": A2C,
        "TD3": TD3,
        "TRPO": TRPO
    }
    
    # Train all baselines
    for name, cls in algorithms.items():
        train_algorithm(name, cls, total_timesteps=1000)
