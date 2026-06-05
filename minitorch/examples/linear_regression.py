from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from minitorch.nn.linear import Linear
from minitorch.nn.losses import mse_loss
from minitorch.optim.adam import Adam
from minitorch.tensor.tensor import Tensor


def main() -> None:
    np.random.seed(7)
    x_values = np.linspace(-2, 2, 64).reshape(-1, 1)
    y_values = 2 * x_values + 3

    x = Tensor(x_values)
    y = Tensor(y_values)

    model = Linear(1, 1)
    optimizer = Adam(model.parameters(), lr=0.05)

    for step in range(1, 801):
        pred = model(x)
        loss = mse_loss(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % 200 == 0:
            print(f"step={step:04d} loss={loss.item():.6f}")

    print("learned weight:", model.weight.data.ravel())
    print("learned bias:", model.bias.data.ravel())


if __name__ == "__main__":
    main()
