from __future__ import annotations

import numpy as np

from tensor.tensor import Tensor
from .module import Module


class LayerNorm(Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = Tensor(np.ones((1, normalized_shape)), requires_grad=True, name="ln_gamma")
        self.beta = Tensor(np.zeros((1, normalized_shape)), requires_grad=True, name="ln_beta")

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(axis=-1, keepdims=True)
        variance = ((x - mean) * (x - mean)).mean(axis=-1, keepdims=True)
        normalized = (x - mean) / (variance + self.eps).pow(0.5)
        return normalized * self.gamma + self.beta
