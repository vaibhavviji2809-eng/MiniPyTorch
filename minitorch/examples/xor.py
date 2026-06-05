from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from minitorch.nn.activations import ReLU, Sigmoid
from minitorch.nn.linear import Linear
from minitorch.nn.losses import binary_cross_entropy
from minitorch.nn.module import Sequential
from minitorch.optim.adam import Adam
from minitorch.tensor.tensor import Tensor


def main() -> None:
    np.random.seed(42)
    x = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = Tensor([[0], [1], [1], [0]])

    model = Sequential(
        Linear(2, 8),
        ReLU(),
        Linear(8, 1),
        Sigmoid(),
    )
    optimizer = Adam(model.parameters(), lr=0.1)

    for step in range(1, 3001):
        pred = model(x)
        loss = binary_cross_entropy(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % 500 == 0:
            print(f"step={step:04d} loss={loss.item():.6f}")

    predictions = model(x).data
    print("predictions:")
    print(np.round(predictions, 4))


if __name__ == "__main__":
    main()
