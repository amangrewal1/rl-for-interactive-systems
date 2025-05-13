from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from tqdm import trange

from .agents.base import Agent
from .env import InteractiveSystemEnv
from .reward_shaping import PotentialShaper, _NullShaper


@dataclass
class TrainLog:
    step: List[int] = field(default_factory=list)
    episode_reward: List[float] = field(default_factory=list)
    episode_regret: List[float] = field(default_factory=list)
    episode_shaped: List[float] = field(default_factory=list)
    instant_reward: List[float] = field(default_factory=list)
    optimal_reward: List[float] = field(default_factory=list)


def train(
    agent: Agent,
    env: InteractiveSystemEnv,
    steps: int,
    shaper=None,
    log_every: int = 100,
    progress: bool = True,
    seed: int = 0,
) -> TrainLog:
    shaper = shaper or _NullShaper()
    log = TrainLog()
    obs = env.reset(seed=seed)
    ep_reward = 0.0
    ep_regret = 0.0
    ep_shaped = 0.0
    prev_history = env.history.copy()
    if hasattr(shaper, "reset"):
        shaper.reset()

    it = trange(steps, disable=not progress, desc=type(agent).__name__)
    for step in it:
        a = agent.act(obs, step=step)
        next_obs, r, done, info = env.step(a)
        shape = 0.0
        if isinstance(shaper, PotentialShaper):
            shape = shaper.shape(prev_history, env.history, done)
        r_train = r + shape
        agent.observe(obs, a, r_train, next_obs, done, step=step)

        ep_reward += r
        ep_regret += max(0.0, info["optimal_reward"] - r)
        ep_shaped += shape
        log.instant_reward.append(float(r))
        log.optimal_reward.append(float(info["optimal_reward"]))
        prev_history = env.history.copy()
        obs = next_obs

        if done:
            log.step.append(step)
            log.episode_reward.append(float(ep_reward))
            log.episode_regret.append(float(ep_regret))
            log.episode_shaped.append(float(ep_shaped))
            ep_reward = 0.0
            ep_regret = 0.0
            ep_shaped = 0.0
            obs = env.reset()
            prev_history = env.history.copy()
            if hasattr(shaper, "reset"):
                shaper.reset()
            agent.end_episode()

        if progress and (step + 1) % log_every == 0 and log.episode_reward:
            it.set_postfix(
                ep_r=f"{np.mean(log.episode_reward[-10:]):.2f}",
                regret=f"{np.mean(log.episode_regret[-10:]):.2f}",
            )
    return log
