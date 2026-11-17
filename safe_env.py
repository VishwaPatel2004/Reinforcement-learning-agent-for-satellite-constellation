from env import SatelliteEnv
import numpy as np
import config as cfg


class SafeSatelliteEnv(SatelliteEnv):
    """
    Extension of SatelliteEnv that tracks constraint costs.

    In standard RL:   Agent maximizes  Σ reward_t
    In Safe RL (CMDP): Agent maximizes  Σ reward_t
                        subject to:     (1/T) Σ cost_t ≤ d

    Where:
        cost_t    = collision probability at step t
        d         = COST_LIMIT = 0.01 (from config.py)
        T         = episode length
    """

    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.episode_costs = []

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.episode_costs = []
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        cost = info['collision_prob']
        self.episode_costs.append(cost)

        info['cost']          = cost
        info['avg_ep_cost']   = np.mean(self.episode_costs) if self.episode_costs else 0
        info['constraint_ok'] = info['avg_ep_cost'] <= cfg.COST_LIMIT

        return obs, reward, terminated, truncated, info

    def get_episode_cost_summary(self):
        if not self.episode_costs:
            return {}
        return {
            "avg_cost":      np.mean(self.episode_costs),
            "max_cost":      np.max(self.episode_costs),
            "n_violations":  sum(1 for c in self.episode_costs if c > cfg.COST_LIMIT),
            "constraint_ok": np.mean(self.episode_costs) <= cfg.COST_LIMIT
        }


if __name__ == "__main__":
    print("Testing SafeSatelliteEnv...")
    env = SafeSatelliteEnv(render_mode="human")
    obs, info = env.reset(seed=42)
    print(f"Initial collision prob: {info['collision_prob']:.6f}")

    for step in range(5):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        env.render()
        print(f"  Cost: {info['cost']:.6f} | Avg cost: {info['avg_ep_cost']:.6f} | "
              f"Constraint OK: {info['constraint_ok']}")
        if term or trunc:
            break

    summary = env.get_episode_cost_summary()
    print(f"\nEpisode summary: {summary}")
    print("SafeSatelliteEnv works!")
