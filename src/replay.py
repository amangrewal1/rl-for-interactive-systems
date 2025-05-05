from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, seed: int = 0):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.ptr = 0
        self._rng = np.random.default_rng(seed)

    def push(self, t: Transition) -> None:
        i = self.ptr
        self.obs[i] = t.obs
        self.next_obs[i] = t.next_obs
        self.actions[i] = t.action
        self.rewards[i] = t.reward
        self.dones[i] = float(t.done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = self._rng.integers(0, self.size, size=batch_size)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )

    def __len__(self) -> int:
        return self.size
