import numpy as np

from .base import Agent


class LinUCBAgent(Agent):
    name = "linucb"

    def __init__(self, n_actions: int, context_dim: int, alpha: float = 1.0, reg: float = 1.0, seed: int = 0):
        self.k = n_actions
        self.d = context_dim
        self.alpha = alpha
        self.A = np.stack([reg * np.eye(context_dim) for _ in range(n_actions)])
        self.b = np.zeros((n_actions, context_dim), dtype=np.float64)
        self._rng = np.random.default_rng(seed)

    def _context(self, obs: np.ndarray) -> np.ndarray:
        return obs[: self.d].astype(np.float64)

    def act(self, obs: np.ndarray, step: int = 0, greedy: bool = False) -> int:
        x = self._context(obs)
        scores = np.zeros(self.k)
        for a in range(self.k):
            Ainv = np.linalg.inv(self.A[a])
            theta = Ainv @ self.b[a]
            mu = float(theta @ x)
            sigma = float(np.sqrt(x @ Ainv @ x))
            scores[a] = mu if greedy else mu + self.alpha * sigma
        return int(np.argmax(scores))

    def observe(self, obs, action, reward, next_obs, done, step: int = 0) -> dict:
        x = self._context(obs)
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x
        return {}


class ThompsonSamplingAgent(Agent):
    name = "thompson"

    def __init__(self, n_actions: int, context_dim: int, v2: float = 0.25, reg: float = 1.0, seed: int = 0):
        self.k = n_actions
        self.d = context_dim
        self.v2 = v2
        self.A = np.stack([reg * np.eye(context_dim) for _ in range(n_actions)])
        self.b = np.zeros((n_actions, context_dim), dtype=np.float64)
        self._rng = np.random.default_rng(seed)

    def _context(self, obs: np.ndarray) -> np.ndarray:
        return obs[: self.d].astype(np.float64)

    def act(self, obs: np.ndarray, step: int = 0, greedy: bool = False) -> int:
        x = self._context(obs)
        scores = np.zeros(self.k)
        for a in range(self.k):
            Ainv = np.linalg.inv(self.A[a])
            mu = Ainv @ self.b[a]
            if greedy:
                scores[a] = mu @ x
            else:
                cov = self.v2 * Ainv
                theta = self._rng.multivariate_normal(mu, cov)
                scores[a] = theta @ x
        return int(np.argmax(scores))

    def observe(self, obs, action, reward, next_obs, done, step: int = 0) -> dict:
        x = self._context(obs)
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x
        return {}
