"""Stub training script for the sequence behavior model (next-app predictor).

This script sketches the pipeline; you can later plug in real data and training code.
"""

from pathlib import Path
from typing import List

import torch
from torch import nn


class SimpleNextAppModel(nn.Module):
    def __init__(self, num_apps: int, embed_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.embed = nn.Embedding(num_apps, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_apps)

    def forward(self, x):  # x: (batch, seq_len)
        emb = self.embed(x)
        out, _ = self.lstm(emb)
        logits = self.fc(out[:, -1, :])
        return logits


def load_sequences() -> List[List[int]]:
    """TODO: load app ID sequences from exported event data."""
    return []


def main() -> None:
    sequences = load_sequences()
    if not sequences:
        print("No data yet; export MLP logs first.")
        return
    # TODO: implement full training loop, evaluation, and ONNX export.


if __name__ == "__main__":
    main()
