import argparse
from itertools import product
from pathlib import Path

import numpy as np

from src.evaluation import evaluate
from src.human_loop import SimulatedHuman, collect_session
from src.train import train
from src.utils import ensure_dir, save_json, set_seed

from .common import build_agent, build_env, build_shaper, default_config


AGENT_STEPS = {"linucb": 10_000, "thompson": 10_000, "dqn": 20_000, "ppo": 30_000}


def run_one(agent_name: str, exploration: str, shaping: str, seed: int, steps: int, cfg: dict):
    set_seed(seed)
    env = build_env(cfg, seed=seed)
    agent = build_agent(agent_name, env, cfg, exploration=exploration, seed=seed)
    shaper = build_shaper(shaping, env, cfg)
    log = train(agent, env, steps=steps, shaper=shaper, seed=seed, progress=False)
    metrics = evaluate(env, lambda o: agent.act(o, greedy=True), n_episodes=20, seed=seed + 10_000)
    rater = SimulatedHuman(seed=seed)
    human_scores = []
    for i in range(5):
        rec = collect_session(env, lambda o: agent.act(o, greedy=True), seed=seed + 20_000 + i)
        human_scores.append(rater.rate(rec).score)
    metrics["human_score"] = float(np.mean(human_scores))
    return {"metrics": metrics, "episode_reward": log.episode_reward, "episode_regret": log.episode_regret}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--steps", type=int, default=None, help="Override per-agent step budget")
    p.add_argument("--out", default="results/bench.json")
    p.add_argument("--agents", nargs="+", default=["linucb", "thompson", "dqn", "ppo"])
    p.add_argument("--explorations", nargs="+", default=["eps_greedy", "boltzmann"])
    p.add_argument("--shapings", nargs="+", default=["none", "diversity"])
    args = p.parse_args()

    cfg = default_config()
    ensure_dir(Path(args.out).parent)

    results = {}
    for agent_name in args.agents:
        explorations = args.explorations if agent_name == "dqn" else ["eps_greedy"]
        for expl, shp in product(explorations, args.shapings):
            key = f"{agent_name}|{expl}|{shp}"
            steps = args.steps or AGENT_STEPS.get(agent_name, 20_000)
            print(f"== {key} x{args.seeds} seeds x{steps} steps ==")
            runs = [run_one(agent_name, expl, shp, seed=s, steps=steps, cfg=cfg) for s in range(args.seeds)]
            agg = {
                "mean_reward": float(np.mean([r["metrics"]["mean_reward"] for r in runs])),
                "std_reward": float(np.std([r["metrics"]["mean_reward"] for r in runs])),
                "mean_regret": float(np.mean([r["metrics"]["mean_regret"] for r in runs])),
                "mean_diversity": float(np.mean([r["metrics"]["mean_diversity"] for r in runs])),
                "optimal_rate": float(np.mean([r["metrics"]["optimal_rate"] for r in runs])),
                "human_score": float(np.mean([r["metrics"]["human_score"] for r in runs])),
            }
            results[key] = {"runs": runs, "aggregate": agg}
            print(f"  -> {agg}")

    save_json({"config": cfg, "results": results, "args": vars(args)}, args.out)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
