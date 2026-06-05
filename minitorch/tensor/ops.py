from __future__ import annotations

import numpy as np


def ensure_array(data) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data.astype(np.float64, copy=False)
    return np.array(data, dtype=np.float64)


def reduce_broadcasted_grad(grad: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    if grad.shape == target_shape:
        return grad

    while len(grad.shape) > len(target_shape):
        grad = grad.sum(axis=0)

    for axis, size in enumerate(target_shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad.reshape(target_shape)
