from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import torch
except ImportError:
    torch = None


@dataclass
class EpsilonSchedule:
    start: float = 1.0
    end: float = 0.05
    decay_steps: int = 20000

    def value(self, step: int) -> float:
        frac = min(1.0, step / max(1, self.decay_steps))
        return float(self.start + (self.end - self.start) * frac)


class EpsilonGreedy:
    def __init__(self, schedule: EpsilonSchedule, n_actions: int, seed: int = 0):
        self.s = schedule
        self.n = n_actions
        self._rng = np.random.default_rng(seed)

    def select(self, q_values: np.ndarray, step: int) -> int:
        if self._rng.random() < self.s.value(step):
            return int(self._rng.integers(0, self.n))
        return int(np.argmax(q_values))


class Boltzmann:
    def __init__(self, temperature: float, seed: int = 0):
        self.tau = max(1e-6, temperature)
        self._rng = np.random.default_rng(seed)

    def select(self, q_values: np.ndarray, step: int = 0) -> int:
        z = q_values / self.tau
        z -= z.max()
        p = np.exp(z)
        p /= p.sum()
        return int(self._rng.choice(len(p), p=p))


class UCB:
    def __init__(self, n_actions: int, c: float = 2.0):
        self.n = n_actions
        self.c = c
        self.counts = np.zeros(n_actions, dtype=np.float64)
        self.total = 0

    def select(self, q_values: np.ndarray, step: Optional[int] = None) -> int:
        if (self.counts == 0).any():
            a = int(np.argmin(self.counts))
        else:
            bonus = self.c * np.sqrt(np.log(self.total + 1) / self.counts)
            a = int(np.argmax(q_values + bonus))
        self.counts[a] += 1
        self.total += 1
        return a

    def reset(self):
        self.counts[:] = 0
        self.total = 0


def entropy_bonus(logits):
    if torch is None:
        raise RuntimeError("torch is required for entropy_bonus")
    logp = torch.log_softmax(logits, dim=-1)
    p = logp.exp()
    return -(p * logp).sum(dim=-1).mean()


def make_explorer(name: str, n_actions: int, cfg: dict, seed: int = 0):
    name = name.lower()
    if name == "eps_greedy":
        s = EpsilonSchedule(
            start=cfg.get("eps_start", 1.0),
            end=cfg.get("eps_end", 0.05),
            decay_steps=cfg.get("eps_decay_steps", 20000),
        )
        return EpsilonGreedy(s, n_actions, seed=seed)
    if name == "boltzmann":
        return Boltzmann(cfg.get("boltzmann_temp", 1.0), seed=seed)
    if name == "ucb":
        return UCB(n_actions, c=cfg.get("ucb_c", 2.0))
    raise ValueError(f"Unknown explorer: {name}")
