"""Very small training harness that runs random policy rollouts and saves a summary."""
from pathlib import Path
from ml.rl_agent.env import SimpleUsageEnv, random_rollout
import json


def run_smoke_training(out_dir: str = 'ml/models/rl'):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    env = SimpleUsageEnv()
    rews = random_rollout(env, steps=50)
    stats = {'mean_reward': float(sum(rews) / max(1, len(rews))), 'steps': len(rews)}
    with open(out / 'rl_smoke.metrics.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f)
    print('Saved RL smoke metrics to', out / 'rl_smoke.metrics.json')


if __name__ == '__main__':
    run_smoke_training()
"""Stub for training PPO/DQN policies for LifeTwin automation."""

from .env import LifeTwinEnv


def train():
    env = LifeTwinEnv()
    # TODO: integrate with stable-baselines3 or custom PPO/DQN implementation.
    # This function should load data, run training, and export a small policy model.
    print("RL training stub; integrate PPO/DQN here.")


if __name__ == "__main__":
    train()
