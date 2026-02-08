# Agents

- `LinUCB` — contextual linear bandit; fast, closed-form updates
- `ThompsonSampling` — linear-Gaussian Thompson; probabilistic
- `DQN` — deep Q-learning with experience replay
- `PPO` — proximal policy optimization with clipped surrogate

Use bandits for low-dimensional context and shallow horizons. Use DQN/PPO
when state dynamics and longer horizons matter.
