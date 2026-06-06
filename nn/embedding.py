from __future__ import annotations

import numpy as np

from tensor.tensor import Tensor
from .module import Module


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Tensor(
            np.random.randn(num_embeddings, embedding_dim) * 0.02,
            requires_grad=True,
            name="embedding_weight",
        )

    def forward(self, token_ids) -> Tensor:
        indices = np.array(token_ids, dtype=np.int64)
        out = Tensor(
            self.weight.data[indices],
            requires_grad=self.weight.requires_grad,
            parents=(self.weight,),
            op="embedding",
        )

        def _backward() -> None:
            if out.grad is None or not self.weight.requires_grad:
                return
            grad_weight = np.zeros_like(self.weight.data)
            np.add.at(grad_weight, indices, out.grad)
            self.weight._add_grad(grad_weight)

        out._backward = _backward
        return out
