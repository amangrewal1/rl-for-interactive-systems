# Exploration Strategies

- ε-greedy — classic, ε decayed on a schedule
- Boltzmann — softmax over Q-values with temperature τ
- UCB — upper confidence bound, log(t) bonus per action
- Entropy bonus — added to PPO loss to maintain action-distribution spread
