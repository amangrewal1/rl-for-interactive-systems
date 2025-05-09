from .base import Agent
from .bandits import LinUCBAgent, ThompsonSamplingAgent

__all__ = ["Agent", "LinUCBAgent", "ThompsonSamplingAgent", "DQNAgent", "PPOAgent"]


def __getattr__(name):
    if name == "DQNAgent":
        from .dqn import DQNAgent
        return DQNAgent
    if name == "PPOAgent":
        from .ppo import PPOAgent
        return PPOAgent
    raise AttributeError(name)
