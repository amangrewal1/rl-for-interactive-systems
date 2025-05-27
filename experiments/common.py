from dataclasses import asdict
from pathlib import Path
from typing import Dict

from src.agents import DQNAgent, LinUCBAgent, PPOAgent, ThompsonSamplingAgent
from src.env import EnvConfig, InteractiveSystemEnv
from src.reward_shaping import make_shaper
from src.utils import device, load_config


def build_env(cfg: Dict, seed: int = 0) -> InteractiveSystemEnv:
    e = cfg["env"]
    return InteractiveSystemEnv(cfg=EnvConfig(**e), seed=seed)


def build_agent(name: str, env: InteractiveSystemEnv, cfg: Dict, exploration: str, seed: int):
    name = name.lower()
    if name == "linucb":
        return LinUCBAgent(env.n_items, env.cfg.context_dim, alpha=cfg["bandit"]["alpha"], reg=cfg["bandit"]["reg"], seed=seed)
    if name == "thompson":
        return ThompsonSamplingAgent(env.n_items, env.cfg.context_dim, reg=cfg["bandit"]["reg"], seed=seed)
    if name == "dqn":
        d = cfg["dqn"]
        return DQNAgent(
            obs_dim=env.obs_dim,
            n_actions=env.n_items,
            hidden=d["hidden"],
            gamma=d["gamma"],
            lr=d["lr"],
            batch_size=d["batch_size"],
            buffer_size=d["buffer_size"],
            target_update=d["target_update"],
            warmup=d["warmup"],
            train_every=d["train_every"],
            exploration=exploration,
            exploration_cfg=cfg["exploration"],
            device=device(),
            seed=seed,
        )
    if name == "ppo":
        p = cfg["ppo"]
        return PPOAgent(
            obs_dim=env.obs_dim,
            n_actions=env.n_items,
            hidden=p["hidden"],
            gamma=p["gamma"],
            gae_lambda=p["gae_lambda"],
            clip_eps=p["clip_eps"],
            lr=p["lr"],
            epochs=p["epochs"],
            batch_size=p["batch_size"],
            rollout=p["rollout"],
            ent_coef=p["ent_coef"],
            vf_coef=p["vf_coef"],
            max_grad_norm=p["max_grad_norm"],
            device=device(),
            seed=seed,
        )
    raise ValueError(f"Unknown agent: {name}")


def build_shaper(name: str, env: InteractiveSystemEnv, cfg: Dict):
    return make_shaper(name, env.n_items, cfg["shaping"])


def default_config() -> Dict:
    return load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
