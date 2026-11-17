import os
import time
from datetime import datetime
import numpy as np
import torch

import config as cfg
from marl_env import MARLSatelliteEnv
from marl_policy import SafeMAPPO


def train_marl(
    total_env_steps=6000,
    rollout_len=256,
    n_epochs=8,
    batch_size=256,
    lr=3e-4,
    seed=42
):
    print("=" * 65)
    print("MULTI-AGENT REINFORCEMENT LEARNING TRAINING (Self Agent Architecture)")
    print("=" * 65)
    print(f"Number of Autonomous Satellites : {cfg.N_SATELLITES}")
    print(f"Debris Catalog Tracked          : {cfg.MAX_DEBRIS}")
    print(f"Local Observation Dimension     : 24 (O(1) invariant)")
    print(f"Local Action Dimension          : 3 ([dvx, dvy, dvz])")
    print(f"Target Total Env Steps          : {total_env_steps:,}")
    print(f"Total Multi-Agent Transitions   : {total_env_steps * cfg.N_SATELLITES:,}")
    print(f"Safety Constraint (Cost Limit)  : {cfg.COST_LIMIT}")
    print("=" * 65)

    np.random.seed(seed)
    torch.manual_seed(seed)

    run_id   = f"marl_mappo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir = os.path.join(cfg.MODELS_DIR, "marl_mappo")
    os.makedirs(save_dir, exist_ok=True)

    env = MARLSatelliteEnv()
    mappo = SafeMAPPO(
        obs_dim=24,
        act_dim=3,
        max_action=0.01,
        lr=lr,
        gamma=cfg.GAMMA,
        gae_lambda=0.95,
        clip_coef=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        device=env.device
    )

    n_sats = env.n_satellites
    obs_batch, _ = env.reset_batched(seed=seed)

    n_rollouts = int(np.ceil(total_env_steps / rollout_len))
    global_env_step = 0
    start_time = time.time()

    best_reward = -float("inf")
    final_model_path = os.path.join(save_dir, "mappo_final_model.pt")

    for r in range(n_rollouts):
        # Buffers for rollout
        mb_obs      = np.zeros((rollout_len, n_sats, 24), dtype=np.float32)
        mb_actions  = np.zeros((rollout_len, n_sats, 3), dtype=np.float32)
        mb_logprobs = np.zeros((rollout_len, n_sats), dtype=np.float32)
        mb_rewards  = np.zeros((rollout_len, n_sats), dtype=np.float32)
        mb_values   = np.zeros((rollout_len, n_sats), dtype=np.float32)
        mb_dones    = np.zeros((rollout_len, n_sats), dtype=np.float32)
        mb_costs    = []

        metrics_min_deb_dist = []
        metrics_min_sat_dist = []
        metrics_coverage     = []
        metrics_fuels        = []

        # 1. Experience Collection Phase
        for t in range(rollout_len):
            global_env_step += 1
            obs_t = torch.tensor(obs_batch, dtype=torch.float32, device=env.device)

            with torch.no_grad():
                action_t, logprob_t, value_t = mappo.get_action(obs_t, deterministic=False)

            actions = action_t.cpu().numpy()
            next_obs, rewards, terms, truncs, infos = env.step_batched(actions)

            mb_obs[t]      = obs_batch
            mb_actions[t]  = actions
            mb_logprobs[t] = logprob_t.cpu().numpy()
            mb_rewards[t]  = rewards
            mb_values[t]   = value_t.cpu().numpy()
            dones          = terms | truncs
            mb_dones[t]    = dones.astype(np.float32)

            for info in infos:
                mb_costs.append(info["p_coll"])
                metrics_min_deb_dist.append(info["min_debris_dist"])
                metrics_min_sat_dist.append(info["min_sat_dist"])
                metrics_coverage.append(info["coverage"])
                metrics_fuels.append(info["fuel"])

            if np.any(dones):
                obs_batch, _ = env.reset_batched()
            else:
                obs_batch = next_obs

        # 2. Generalized Advantage Estimation (GAE)
        with torch.no_grad():
            next_obs_t = torch.tensor(obs_batch, dtype=torch.float32, device=env.device)
            next_value = mappo.critic(next_obs_t).squeeze(-1).cpu().numpy()

        mb_advantages = np.zeros_like(mb_rewards)
        last_gae_lam  = np.zeros(n_sats, dtype=np.float32)

        for t in reversed(range(rollout_len)):
            if t == rollout_len - 1:
                next_non_terminal = 1.0 - mb_dones[t]
                next_val          = next_value
            else:
                next_non_terminal = 1.0 - mb_dones[t]
                next_val          = mb_values[t + 1]

            delta = mb_rewards[t] + cfg.GAMMA * next_val * next_non_terminal - mb_values[t]
            last_gae_lam = delta + cfg.GAMMA * 0.95 * next_non_terminal * last_gae_lam
            mb_advantages[t] = last_gae_lam

        mb_returns = mb_advantages + mb_values

        # 3. Flatten Multi-Agent Transitions across Time and Satellites
        b_obs        = mb_obs.reshape(-1, 24)
        b_actions    = mb_actions.reshape(-1, 3)
        b_logprobs   = mb_logprobs.reshape(-1)
        b_returns    = mb_returns.reshape(-1)
        b_advantages = mb_advantages.reshape(-1)

        # 4. Update Policy and Lagrangian Multiplier
        update_info = mappo.update(
            b_obs, b_actions, b_logprobs, b_returns, b_advantages,
            b_costs=mb_costs, n_epochs=n_epochs, batch_size=batch_size
        )

        mean_ep_reward = float(np.mean(mb_rewards))
        mean_deb_dist  = float(np.mean(metrics_min_deb_dist))
        min_deb_dist   = float(np.min(metrics_min_deb_dist))
        mean_sat_dist  = float(np.mean(metrics_min_sat_dist))
        min_sat_dist   = float(np.min(metrics_min_sat_dist))
        mean_p_coll    = float(np.mean(mb_costs))
        mean_fuel      = float(np.mean(metrics_fuels))
        mean_cov       = float(np.mean(metrics_coverage))

        elapsed = time.time() - start_time
        fps = (global_env_step * n_sats) / max(0.001, elapsed)

        print(
            f"Iter [{r+1:2d}/{n_rollouts:2d}] | "
            f"EnvSteps: {global_env_step:5d} | "
            f"Rew: {mean_ep_reward:6.2f} | "
            f"DebMin: {min_deb_dist:6.1f}km | "
            f"SatMin: {min_sat_dist:6.1f}km | "
            f"Pc: {mean_p_coll:.5f} | "
            f"Lambda: {update_info['lambda']:.3f} | "
            f"Cov: {mean_cov:.1%} | "
            f"Fuel: {mean_fuel:.1%} | "
            f"FPS: {fps:.0f}"
        )

        if mean_ep_reward > best_reward:
            best_reward = mean_ep_reward
            mappo.save(os.path.join(save_dir, "best_mappo_model.pt"))

    # Save final model
    mappo.save(final_model_path)

    # Write reference file
    ref_file = os.path.join(cfg.MODELS_DIR, "latest_marl.txt")
    with open(ref_file, "w") as f:
        f.write(run_id + "\n" + final_model_path + "\n")

    print("\n" + "=" * 65)
    print("MARL TRAINING COMPLETE!")
    print(f"Model saved to: {final_model_path}")
    print(f"Reference file: {ref_file}")
    print(f"Total time: {time.time() - start_time:.1f}s")
    print("=" * 65)
    return final_model_path


if __name__ == "__main__":
    train_marl()
