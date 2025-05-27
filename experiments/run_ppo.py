import argparse

from src.evaluation import evaluate
from src.train import train
from src.utils import save_json, set_seed

from .common import build_agent, build_env, build_shaper, default_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shaping", default="none", choices=["none", "diversity", "novelty", "both"])
    p.add_argument("--out", default=None)
    p.add_argument("--checkpoint", default=None)
    args = p.parse_args()

    cfg = default_config()
    set_seed(args.seed)
    env = build_env(cfg, seed=args.seed)
    agent = build_agent("ppo", env, cfg, exploration="eps_greedy", seed=args.seed)
    shaper = build_shaper(args.shaping, env, cfg)

    log = train(agent, env, steps=args.steps, shaper=shaper, seed=args.seed)
    metrics = evaluate(env, lambda o: agent.act(o, greedy=True), n_episodes=20, seed=args.seed + 10_000)
    print("Evaluation:", metrics)

    if args.checkpoint:
        agent.save(args.checkpoint)
    if args.out:
        save_json(
            {"args": vars(args), "metrics": metrics, "log": log.__dict__},
            args.out,
        )


if __name__ == "__main__":
    main()
