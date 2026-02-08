# Reward Shaping

Potential-based shaping preserves optimality under standard regularity
conditions. Two variants implemented:

- **Diversity shaping** — potential = -entropy of recent action
  distribution. Encourages the policy to vary its picks.
- **Novelty shaping** — potential = visitation-count bonus over
  discretized state cells.
