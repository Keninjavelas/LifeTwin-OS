"""Evaluation stub for next-app prediction model.

Computes top-1/top-3 accuracy once real data and a trained model are available.
"""

from typing import List, Tuple

import torch


def compute_topk(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    topk = logits.topk(k, dim=-1).indices
    correct = (topk == targets.unsqueeze(-1)).any(dim=-1).float().mean().item()
    return correct


def evaluate(model, dataloader) -> Tuple[float, float]:
    # TODO: load model and dataset
    top1 = 0.0
    top3 = 0.0
    return top1, top3
