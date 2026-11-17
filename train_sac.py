import os
import numpy as np
from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from env import SatelliteEnv
import config as cfg

if __name__ == "__main__":
    print("=" * 60)
    print("TRAINING BASELINE: SAC")
    print("=" * 60)
    
    run_id = f"sac_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir = os.path.join(cfg.MODELS_DIR, "baselines", run_id)
    os.makedirs(save_dir, exist_ok=True)
    
    env = DummyVecEnv([lambda: Monitor(SatelliteEnv())])
    
    model = SAC(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=cfg.SEED,
        buffer_size=15000,
        learning_starts=500
    )
    
    print(f"Starting training for SAC (1000 steps)...")
    try:
        model.learn(total_timesteps=1000, progress_bar=True)
    except KeyboardInterrupt:
        print("\nTraining stopped early.")
        
    final_path = os.path.join(save_dir, "sac_final_model")
    model.save(final_path)
    
    os.makedirs(os.path.join(cfg.MODELS_DIR, "baselines"), exist_ok=True)
    with open(os.path.join(cfg.MODELS_DIR, "baselines", "latest_sac.txt"), 'w') as f:
        f.write(run_id + "\n" + final_path + ".zip")
        
    print(f"SAC training complete and saved to {final_path}.zip\n")
