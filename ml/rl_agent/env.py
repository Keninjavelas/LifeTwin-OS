"""Minimal RL environment scaffold.

Provides a tiny Gym-like interface for prototyping policy training.
"""
from typing import Tuple, List
import random


class SimpleUsageEnv:
    """A toy environment where state is a single scalar (screen minutes).

    Actions: 0 (do nothing), 1 (nudge reduce usage)
    Reward: negative of screen minutes (we try to minimize screen time)
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.state = 60

    def reset(self) -> int:
        self.state = random.randint(30, 120)
        return self.state

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        # simple dynamics
        if action == 1:
            self.state = max(0, self.state - random.randint(1, 10))
        else:
            self.state = self.state + random.randint(0, 5)
        reward = -float(self.state)
        done = False
        return self.state, reward, done, {}


def random_rollout(env: SimpleUsageEnv, steps: int = 10) -> List[float]:
    s = env.reset()
    rews = []
    for _ in range(steps):
        a = random.choice([0, 1])
        s, r, done, _ = env.step(a)
        rews.append(r)
    return rews


if __name__ == '__main__':
    e = SimpleUsageEnv()
    print(random_rollout(e, 20))
"""Gym-style environment stub for RL-based automation policy."""

from typing import Any, Dict, Tuple


class LifeTwinEnv:
    def __init__(self):
        # TODO: define observation/action spaces using gymnasium or similar
        self.state = None

    def reset(self) -> Any:
        # TODO: reset environment state
        self.state = {}
        return self.state

    def step(self, action: int) -> Tuple[Any, float, bool, Dict]:
        # TODO: apply action, compute reward, and transition state
        next_state = self.state
        reward = 0.0
        done = False
        info: Dict = {}
        return next_state, reward, done, info
