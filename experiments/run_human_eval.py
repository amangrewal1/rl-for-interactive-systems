import argparse

from src.human_loop import SimulatedHuman, cli_rate, collect_session
from src.utils import set_seed

from .common import build_agent, build_env, default_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", choices=["linucb", "thompson", "dqn", "ppo"], default="dqn")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--simulated", action="store_true", help="Use simulated rater instead of CLI")
    p.add_argument("--diversity_weight", type=float, default=0.4)
    args = p.parse_args()

    cfg = default_config()
    set_seed(args.seed)
    env = build_env(cfg, seed=args.seed)
    agent = build_agent(args.agent, env, cfg, exploration="eps_greedy", seed=args.seed)
    if args.checkpoint and hasattr(agent, "load"):
        agent.load(args.checkpoint)

    rater = SimulatedHuman(diversity_weight=args.diversity_weight, seed=args.seed)
    scores = []
    for i in range(args.episodes):
        rec = collect_session(env, lambda o: agent.act(o, greedy=True), seed=args.seed + 1000 + i)
        rating = rater.rate(rec) if args.simulated else cli_rate(rec)
        print(f"Episode {i}: score={rating.score:.3f} ({rating.notes})")
        scores.append(rating.score)

    scores = [s for s in scores if s == s]
    if scores:
        print(f"\nMean human score over {len(scores)} sessions: {sum(scores)/len(scores):.3f}")


if __name__ == "__main__":
    main()
