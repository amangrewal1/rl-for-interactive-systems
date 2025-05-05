from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class EnvConfig:
    n_items: int = 10
    context_dim: int = 16
    horizon: int = 200
    satiation_rate: float = 0.35
    recovery_rate: float = 0.05
    drift_scale: float = 0.01
    noise_std: float = 0.1
    history_len: int = 8


@dataclass
class StepInfo:
    true_best: int
    optimal_reward: float
    user_pref: np.ndarray = field(repr=False)
    satiation: np.ndarray = field(repr=False)


class InteractiveSystemEnv:
    """Contextual, non-stationary recommender environment.

    A simulated user has latent preferences over `n_items` content categories.
    The agent picks one item per step; reward is engagement shaped by (i) the
    user's current affinity for the chosen item, (ii) item-specific satiation
    that grows with repeated exposure and recovers slowly otherwise, and
    (iii) slow random drift in preferences. The observation concatenates the
    static user embedding, a recent-action histogram, and a time feature.
    """

    def __init__(self, cfg: Optional[EnvConfig] = None, seed: Optional[int] = None):
        self.cfg = cfg or EnvConfig()
        self._rng = np.random.default_rng(seed)
        self._init_items()
        self.t = 0
        self.user_pref = None
        self.user_embed = None
        self.satiation = None
        self.history = None

    @property
    def n_items(self) -> int:
        return self.cfg.n_items

    @property
    def obs_dim(self) -> int:
        return self.cfg.context_dim + self.cfg.n_items + 1

    def _init_items(self):
        d = self.cfg.context_dim
        k = self.cfg.n_items
        M = self._rng.normal(size=(k, d)).astype(np.float32)
        M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-8
        self.item_embed = M

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        d = self.cfg.context_dim
        k = self.cfg.n_items
        self.user_embed = self._rng.normal(scale=0.5, size=(d,)).astype(np.float32)
        self.user_pref = (self.item_embed @ self.user_embed).astype(np.float32)
        self.satiation = np.zeros(k, dtype=np.float32)
        self.history = np.zeros(k, dtype=np.float32)
        self.t = 0
        return self._obs()

    def _obs(self) -> np.ndarray:
        hist_norm = self.history / max(self.history.sum(), 1.0)
        t_feat = np.array([self.t / self.cfg.horizon], dtype=np.float32)
        return np.concatenate([self.user_embed, hist_norm, t_feat]).astype(np.float32)

    def optimal_action(self) -> tuple[int, float]:
        effective = self.user_pref * (1.0 - self.satiation)
        a = int(np.argmax(effective))
        return a, float(effective[a])

    def step(self, action: int):
        assert 0 <= action < self.cfg.n_items
        best, best_r = self.optimal_action()
        pref = float(self.user_pref[action])
        sat = float(self.satiation[action])
        signal = pref * (1.0 - sat)
        noise = float(self._rng.normal(scale=self.cfg.noise_std))
        reward = signal + noise

        self.satiation *= (1.0 - self.cfg.recovery_rate)
        self.satiation[action] = min(1.0, self.satiation[action] + self.cfg.satiation_rate)
        drift = self._rng.normal(scale=self.cfg.drift_scale, size=self.user_embed.shape).astype(np.float32)
        self.user_embed = self.user_embed + drift
        self.user_pref = (self.item_embed @ self.user_embed).astype(np.float32)
        self.history[action] += 1.0
        self.t += 1

        done = self.t >= self.cfg.horizon
        info = {"optimal_action": best, "optimal_reward": best_r, "true_signal": signal}
        return self._obs(), float(reward), bool(done), info

    def snapshot(self) -> StepInfo:
        a, r = self.optimal_action()
        return StepInfo(true_best=a, optimal_reward=r, user_pref=self.user_pref.copy(), satiation=self.satiation.copy())
