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
