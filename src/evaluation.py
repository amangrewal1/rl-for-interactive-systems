from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .env import InteractiveSystemEnv


@dataclass
class EpisodeStats:
    total_reward: float = 0.0
    regret: float = 0.0
    actions: List[int] = field(default_factory=list)
    optimal_actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    length: int = 0

    def diversity(self) -> float:
        if not self.actions:
            return 0.0
        counts = np.bincount(self.actions, minlength=max(self.actions) + 1).astype(np.float64)
        p = counts / counts.sum()
        p = np.clip(p, 1e-12, 1.0)
        return float(-(p * np.log(p)).sum())

    def optimal_rate(self) -> float:
        if not self.actions:
            return 0.0
        return float(np.mean(np.array(self.actions) == np.array(self.optimal_actions)))


def evaluate(
    env: InteractiveSystemEnv,
    policy: Callable[[np.ndarray], int],
    n_episodes: int = 10,
    seed: int = 0,
) -> Dict[str, float]:
    rng_seed = seed
    rewards, regrets, divs, opt_rates, lens = [], [], [], [], []
    for e in range(n_episodes):
        obs = env.reset(seed=rng_seed + e)
        stats = EpisodeStats()
        done = False
        while not done:
            a = int(policy(obs))
            info_snap = env.snapshot()
            obs, r, done, info = env.step(a)
            stats.total_reward += r
            stats.actions.append(a)
            stats.optimal_actions.append(info["optimal_action"])
            stats.rewards.append(r)
            stats.regret += max(0.0, info["optimal_reward"] - r)
            stats.length += 1
        rewards.append(stats.total_reward)
        regrets.append(stats.regret)
        divs.append(stats.diversity())
        opt_rates.append(stats.optimal_rate())
        lens.append(stats.length)
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_regret": float(np.mean(regrets)),
        "mean_diversity": float(np.mean(divs)),
        "optimal_rate": float(np.mean(opt_rates)),
        "mean_length": float(np.mean(lens)),
    }


def cumulative_regret(rewards: List[float], optimal_rewards: List[float]) -> np.ndarray:
    r = np.asarray(rewards, dtype=np.float32)
    o = np.asarray(optimal_rewards, dtype=np.float32)
    return np.cumsum(np.maximum(o - r, 0.0))


def summarize_runs(runs: List[Dict[str, List[float]]], key: str) -> Dict[str, np.ndarray]:
    xs = np.stack([np.asarray(r[key], dtype=np.float32) for r in runs])
    return {
        "mean": xs.mean(axis=0),
        "std": xs.std(axis=0),
        "min": xs.min(axis=0),
        "max": xs.max(axis=0),
    }
