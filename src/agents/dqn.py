from copy import deepcopy
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..exploration import make_explorer
from ..replay import ReplayBuffer, Transition
from .base import Agent


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: List[int]):
        super().__init__()
        dims = [in_dim, *hidden]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DQNAgent(Agent):
    name = "dqn"

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: List[int] = (128, 128),
        gamma: float = 0.95,
        lr: float = 3e-4,
        batch_size: int = 128,
        buffer_size: int = 50000,
        target_update: int = 500,
        warmup: int = 1000,
        train_every: int = 4,
        exploration: str = "eps_greedy",
        exploration_cfg: dict | None = None,
        device: torch.device | None = None,
        seed: int = 0,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.warmup = warmup
        self.train_every = train_every
        self.device = device or torch.device("cpu")
        torch.manual_seed(seed)

        self.q = MLP(obs_dim, n_actions, list(hidden)).to(self.device)
        self.q_target = deepcopy(self.q).to(self.device)
        for p in self.q_target.parameters():
            p.requires_grad_(False)

        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size, obs_dim, seed=seed)
        self.explorer = make_explorer(exploration, n_actions, exploration_cfg or {}, seed=seed)
        self._updates = 0

    @torch.no_grad()
    def _q_values(self, obs: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        return self.q(t).cpu().numpy().squeeze(0)

    def act(self, obs: np.ndarray, step: int = 0, greedy: bool = False) -> int:
        qv = self._q_values(obs)
        if greedy:
            return int(np.argmax(qv))
        return int(self.explorer.select(qv, step=step))

    def observe(self, obs, action, reward, next_obs, done, step: int = 0) -> dict:
        self.buffer.push(Transition(obs, action, reward, next_obs, done))
        if len(self.buffer) < max(self.warmup, self.batch_size):
            return {}
        if step % self.train_every != 0:
            return {}
        return self._train_step()

    def _train_step(self) -> dict:
        o, a, r, op, d = self.buffer.sample(self.batch_size)
        o = torch.from_numpy(o).to(self.device)
        a = torch.from_numpy(a).to(self.device)
        r = torch.from_numpy(r).to(self.device)
        op = torch.from_numpy(op).to(self.device)
        d = torch.from_numpy(d).to(self.device)

        q_sa = self.q(o).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            a_next = self.q(op).argmax(dim=1, keepdim=True)
            q_next = self.q_target(op).gather(1, a_next).squeeze(1)
            target = r + self.gamma * (1.0 - d) * q_next

        loss = F.smooth_l1_loss(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()

        self._updates += 1
        if self._updates % self.target_update == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        return {"q_loss": float(loss.item()), "q_mean": float(q_sa.mean().item())}

    def save(self, path: str) -> None:
        torch.save({"q": self.q.state_dict()}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.q.load_state_dict(ckpt["q"])
        self.q_target.load_state_dict(ckpt["q"])
