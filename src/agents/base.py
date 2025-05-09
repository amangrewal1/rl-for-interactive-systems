from abc import ABC, abstractmethod

import numpy as np


class Agent(ABC):
    name: str = "agent"

    @abstractmethod
    def act(self, obs: np.ndarray, step: int = 0, greedy: bool = False) -> int: ...

    def observe(self, obs, action, reward, next_obs, done, step: int = 0) -> dict:
        return {}

    def end_episode(self) -> None: pass

    def save(self, path: str) -> None: pass

    def load(self, path: str) -> None: pass
