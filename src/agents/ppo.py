from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .base import Agent
from .dqn import MLP


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: List[int]):
        super().__init__()
        self.shared = MLP(obs_dim, hidden[-1], list(hidden[:-1]) if len(hidden) > 1 else [hidden[0]])
        self.pi = nn.Linear(hidden[-1], n_actions)
        self.v = nn.Linear(hidden[-1], 1)

    def forward(self, x):
        h = self.shared(x)
        return self.pi(h), self.v(h).squeeze(-1)


class RolloutBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.logp = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.capacity = capacity

    def add(self, o, a, lp, r, v, d):
        i = self.ptr
        self.obs[i] = o
        self.actions[i] = a
        self.logp[i] = lp
        self.rewards[i] = r
        self.values[i] = v
        self.dones[i] = float(d)
        self.ptr += 1

    def full(self) -> bool:
        return self.ptr >= self.capacity

    def reset(self):
        self.ptr = 0


class PPOAgent(Agent):
    name = "ppo"

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: List[int] = (128, 128),
        gamma: float = 0.95,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        lr: float = 3e-4,
        epochs: int = 4,
        batch_size: int = 64,
        rollout: int = 1024,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: torch.device | None = None,
        seed: int = 0,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.lam = gae_lambda
        self.clip = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.device = device or torch.device("cpu")
        torch.manual_seed(seed)
        self._rng = np.random.default_rng(seed)

        self.ac = ActorCritic(obs_dim, n_actions, list(hidden)).to(self.device)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=lr)
        self.buf = RolloutBuffer(rollout, obs_dim)
        self._last_next_obs = None

    @torch.no_grad()
    def _policy(self, obs: np.ndarray):
        t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        logits, v = self.ac(t)
        dist = Categorical(logits=logits)
        a = dist.sample()
        return int(a.item()), float(dist.log_prob(a).item()), float(v.item())

    @torch.no_grad()
    def _greedy(self, obs: np.ndarray) -> int:
        t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        logits, _ = self.ac(t)
        return int(torch.argmax(logits, dim=-1).item())

    def act(self, obs: np.ndarray, step: int = 0, greedy: bool = False) -> int:
        if greedy:
            return self._greedy(obs)
        a, lp, v = self._policy(obs)
        self._cache = (obs, a, lp, v)
        return a

    def observe(self, obs, action, reward, next_obs, done, step: int = 0) -> dict:
        o, a, lp, v = self._cache
        self.buf.add(o, a, lp, reward, v, done)
        self._last_next_obs = next_obs
        if self.buf.full():
            return self._update()
        return {}

    def _compute_gae(self, last_value: float):
        T = self.buf.ptr
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            next_v = last_value if t == T - 1 else self.buf.values[t + 1]
            next_nonterminal = 1.0 - self.buf.dones[t]
            delta = self.buf.rewards[t] + self.gamma * next_v * next_nonterminal - self.buf.values[t]
            gae = delta + self.gamma * self.lam * next_nonterminal * gae
            adv[t] = gae
        ret = adv + self.buf.values[:T]
        return adv, ret

    def _update(self) -> dict:
        with torch.no_grad():
            t = torch.from_numpy(self._last_next_obs).float().unsqueeze(0).to(self.device)
            _, last_v = self.ac(t)
            last_v = float(last_v.item())

        adv, ret = self._compute_gae(last_v)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs = torch.from_numpy(self.buf.obs[: self.buf.ptr]).to(self.device)
        actions = torch.from_numpy(self.buf.actions[: self.buf.ptr]).to(self.device)
        old_logp = torch.from_numpy(self.buf.logp[: self.buf.ptr]).to(self.device)
        adv_t = torch.from_numpy(adv).to(self.device)
        ret_t = torch.from_numpy(ret).to(self.device)

        T = self.buf.ptr
        idxs = np.arange(T)
        losses = {"pi": 0.0, "v": 0.0, "ent": 0.0}
        n_batches = 0
        for _ in range(self.epochs):
            self._rng.shuffle(idxs)
            for start in range(0, T, self.batch_size):
                b = idxs[start : start + self.batch_size]
                bi = torch.from_numpy(b).long().to(self.device)
                logits, v = self.ac(obs[bi])
                dist = Categorical(logits=logits)
                logp = dist.log_prob(actions[bi])
                ratio = torch.exp(logp - old_logp[bi])
                a_b = adv_t[bi]
                pg1 = ratio * a_b
                pg2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * a_b
                pi_loss = -torch.min(pg1, pg2).mean()
                v_loss = F.mse_loss(v, ret_t[bi])
                ent = dist.entropy().mean()
                loss = pi_loss + self.vf_coef * v_loss - self.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.opt.step()
                losses["pi"] += float(pi_loss.item())
                losses["v"] += float(v_loss.item())
                losses["ent"] += float(ent.item())
                n_batches += 1

        self.buf.reset()
        for k in losses:
            losses[k] /= max(1, n_batches)
        return {f"ppo_{k}": v for k, v in losses.items()}

    def save(self, path: str) -> None:
        torch.save({"ac": self.ac.state_dict()}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(ckpt["ac"])
