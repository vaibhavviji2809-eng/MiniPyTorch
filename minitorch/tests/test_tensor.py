from __future__ import annotations

import unittest

import numpy as np

from minitorch.tensor.tensor import Tensor
from minitorch.nn.linear import Linear
from minitorch.nn.losses import mse_loss


class TensorAutogradTests(unittest.TestCase):
    def test_scalar_backward(self) -> None:
        x = Tensor(2.0, requires_grad=True)
        y = x * x
        y.backward()
        self.assertAlmostEqual(float(x.grad), 4.0, places=5)

    def test_matmul_backward(self) -> None:
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        w = Tensor([[0.5], [0.8]], requires_grad=True)
        loss = (x @ w).mean()
        loss.backward()
        np.testing.assert_allclose(w.grad, np.array([[2.0], [3.0]]), atol=1e-6)

    def test_linear_learns_simple_mapping(self) -> None:
        np.random.seed(0)
        x = Tensor(np.array([[0.0], [1.0], [2.0], [3.0]]))
        y = Tensor(np.array([[1.0], [3.0], [5.0], [7.0]]))
        layer = Linear(1, 1)

        for _ in range(400):
            pred = layer(x)
            loss = mse_loss(pred, y)
            layer.zero_grad()
            loss.backward()
            for param in layer.parameters():
                param.data = param.data - 0.05 * param.grad

        final_loss = mse_loss(layer(x), y).item()
        self.assertLess(final_loss, 1e-3)


if __name__ == "__main__":
    unittest.main()
