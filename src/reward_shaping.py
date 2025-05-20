from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class ShapingConfig:
    diversity_weight: float = 0.1
    novelty_weight: float = 0.1
    gamma: float = 0.95


class PotentialShaper:
    """Potential-based shaping F(s, s') = gamma * phi(s') - phi(s).

    Preserves optimal policy (Ng et al., 1999). Potentials encourage exploration
    of novel items and diverse action distributions across a session.
    """

    def __init__(self, n_items: int, cfg: ShapingConfig):
        self.n_items = n_items
        self.cfg = cfg
        self.prev_phi = 0.0

    def reset(self) -> None:
        self.prev_phi = 0.0

    def _diversity_potential(self, history: np.ndarray) -> float:
        total = history.sum()
        if total <= 0:
            return 0.0
        p = history / total
        p = np.clip(p, 1e-12, 1.0)
        ent = float(-(p * np.log(p)).sum())
        return ent / np.log(self.n_items)

    def _novelty_potential(self, history: np.ndarray) -> float:
        total = history.sum()
        if total <= 0:
            return 1.0
        return float(1.0 - (history.max() / total))

    def potential(self, history: np.ndarray) -> float:
        d = self._diversity_potential(history)
        n = self._novelty_potential(history)
        return self.cfg.diversity_weight * d + self.cfg.novelty_weight * n

    def shape(self, prev_history: np.ndarray, next_history: np.ndarray, done: bool) -> float:
        phi_s = self.potential(prev_history)
        phi_s_next = 0.0 if done else self.potential(next_history)
        return self.cfg.gamma * phi_s_next - phi_s


def make_shaper(name: str, n_items: int, cfg: dict) -> Callable:
    name = (name or "none").lower()
    if name == "none":
        return _NullShaper()
    sc = ShapingConfig(
        diversity_weight=cfg.get("diversity_weight", 0.1) if name in {"diversity", "both"} else 0.0,
        novelty_weight=cfg.get("novelty_weight", 0.1) if name in {"novelty", "both"} else 0.0,
        gamma=cfg.get("gamma", 0.95),
    )
    return PotentialShaper(n_items, sc)


class _NullShaper:
    def reset(self): pass
    def shape(self, *_args, **_kwargs) -> float: return 0.0
    def potential(self, *_a, **_k) -> float: return 0.0
