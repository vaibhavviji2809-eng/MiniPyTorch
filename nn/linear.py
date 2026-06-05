from __future__ import annotations

import math
import numpy as np

from tensor.tensor import Tensor
from .module import Module


class Linear(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        bound = math.sqrt(2.0 / in_features)
        self.weight = Tensor(
            np.random.randn(in_features, out_features) * bound,
            requires_grad=True,
            name="weight",
        )
        self.bias = Tensor(
            np.zeros((1, out_features)),
            requires_grad=True,
            name="bias",
        )

    def forward(self, x: Tensor) -> Tensor:
        return (x @ self.weight) + self.bias
