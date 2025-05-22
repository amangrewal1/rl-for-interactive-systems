from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from .env import InteractiveSystemEnv


@dataclass
class SessionRecord:
    actions: List[int]
    rewards: List[float]
    optimal_actions: List[int]
    optimal_rewards: List[float]


@dataclass
class HumanRating:
    score: float
    notes: str = ""


class SimulatedHuman:
    """Deterministic proxy rater: rewards high engagement AND high diversity.

    Rationale: real users may score a session well even when cumulative reward
    is modest, provided the experience felt varied. Diversity term approximates
    that qualitative preference without requiring real humans in automated sweeps.
    """

    def __init__(self, diversity_weight: float = 0.4, noise_std: float = 0.05, seed: int = 0):
        self.dw = diversity_weight
        self.noise = noise_std
        self._rng = np.random.default_rng(seed)

    def rate(self, rec: SessionRecord) -> HumanRating:
        r = np.asarray(rec.rewards, dtype=np.float64)
        a = np.asarray(rec.actions)
        n = int(a.max()) + 1 if len(a) else 1
        counts = np.bincount(a, minlength=n).astype(np.float64)
        p = counts / max(counts.sum(), 1.0)
        p = np.clip(p, 1e-12, 1.0)
        ent = float(-(p * np.log(p)).sum()) / np.log(max(n, 2))
        base = float(np.tanh(r.mean()))
        score = (1.0 - self.dw) * base + self.dw * ent
        score = float(np.clip(score + self._rng.normal(scale=self.noise), 0.0, 1.0))
        return HumanRating(score=score, notes=f"engagement={base:.3f} diversity={ent:.3f}")


def collect_session(
    env: InteractiveSystemEnv,
    policy: Callable[[np.ndarray], int],
    seed: Optional[int] = None,
) -> SessionRecord:
    obs = env.reset(seed=seed)
    rec = SessionRecord([], [], [], [])
    done = False
    while not done:
        a = int(policy(obs))
        obs, r, done, info = env.step(a)
        rec.actions.append(a)
        rec.rewards.append(r)
        rec.optimal_actions.append(info["optimal_action"])
        rec.optimal_rewards.append(info["optimal_reward"])
    return rec


def cli_rate(rec: SessionRecord) -> HumanRating:
    print("\n--- Session summary ---")
    print(f"Steps: {len(rec.actions)}")
    print(f"Total reward: {sum(rec.rewards):.2f}")
    print(f"Distinct items: {len(set(rec.actions))}")
    print(f"Most-picked item: {max(set(rec.actions), key=rec.actions.count)}")
    print(f"Optimal-match rate: {np.mean(np.array(rec.actions)==np.array(rec.optimal_actions)):.2%}")
    while True:
        raw = input("\nRate this session 0.0 - 1.0 (or 'skip'): ").strip()
        if raw.lower() == "skip":
            return HumanRating(score=float("nan"), notes="skipped")
        try:
            s = float(raw)
            if 0.0 <= s <= 1.0:
                notes = input("Notes (enter to skip): ").strip()
                return HumanRating(score=s, notes=notes)
        except ValueError:
            pass
        print("Please enter a number between 0.0 and 1.0.")
