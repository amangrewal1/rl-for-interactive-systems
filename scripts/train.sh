#!/usr/bin/env bash
set -euo pipefail

# Train all agents with default config
python -m experiments.run_bandits --agent linucb --steps 10000
python -m experiments.run_dqn --steps 30000 --exploration boltzmann
python -m experiments.run_ppo --steps 50000 --shaping diversity
