from __future__ import annotations

import numpy as np

from tensor.tensor import Tensor


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    target = Tensor.ensure_tensor(target)
    diff = pred - target
    return (diff * diff).mean()


def binary_cross_entropy(pred: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    target = Tensor.ensure_tensor(target)
    pred_clamped = pred * (1 - 2 * eps) + eps
    return -((target * pred_clamped.log()) + ((1 - target) * (1 - pred_clamped).log())).mean()


def cross_entropy(logits: Tensor, target_indices: np.ndarray | list[int]) -> Tensor:
    target_array = np.array(target_indices, dtype=np.int64).reshape(-1)
    shifted = logits - Tensor(logits.data.max(axis=1, keepdims=True))
    exp_scores = shifted.exp()
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    one_hot = np.zeros_like(logits.data)
    one_hot[np.arange(len(target_array)), target_array] = 1.0
    targets = Tensor(one_hot)
    return -(targets * probs.log()).sum(axis=1).mean()
