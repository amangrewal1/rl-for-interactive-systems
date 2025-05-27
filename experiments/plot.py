import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils import ensure_dir, running_mean


def _pad(xs, target_len):
    out = np.full((len(xs), target_len), np.nan, dtype=np.float32)
    for i, x in enumerate(xs):
        x = np.asarray(x, dtype=np.float32)
        out[i, : len(x)] = x
    return out


def curves(results: dict, metric: str, out: Path):
    plt.figure(figsize=(8, 5))
    for key, v in results.items():
        series = [r[metric] for r in v["runs"]]
        max_len = max(len(s) for s in series)
        arr = _pad(series, max_len)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        smoothed = running_mean(mean, 20)
        x = np.arange(len(smoothed))
        plt.plot(x, smoothed, label=key)
        if len(smoothed) == len(mean):
            plt.fill_between(x, smoothed - std[: len(smoothed)], smoothed + std[: len(smoothed)], alpha=0.15)
    plt.xlabel("episode")
    plt.ylabel(metric)
    plt.title(metric.replace("_", " "))
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()


def bar_summary(results: dict, metric: str, out: Path):
    keys = list(results.keys())
    vals = [results[k]["aggregate"][metric] for k in keys]
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(keys)), vals)
    plt.xticks(range(len(keys)), keys, rotation=30, ha="right", fontsize=8)
    plt.ylabel(metric)
    plt.title(f"Final {metric}")
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--out", default="results/figures")
    args = p.parse_args()

    with open(args.path) as f:
        bench = json.load(f)
    results = bench["results"]
    out = ensure_dir(args.out)

    curves(results, "episode_reward", out / "episode_reward.png")
    curves(results, "episode_regret", out / "episode_regret.png")
    for m in ["mean_reward", "mean_regret", "optimal_rate", "mean_diversity", "human_score"]:
        bar_summary(results, m, out / f"bar_{m}.png")
    print(f"Figures written to {out}")


if __name__ == "__main__":
    main()
