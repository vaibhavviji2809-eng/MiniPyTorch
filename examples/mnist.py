from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nn import Conv2D, Linear, MaxPool2D, ReLU, Sequential
from tensor.tensor import Tensor


class MiniConvNet:
    def __init__(self) -> None:
        self.conv1 = Conv2D(1, 4, kernel_size=3, padding=1)
        self.act1 = ReLU()
        self.pool1 = MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = Conv2D(4, 8, kernel_size=3, padding=1)
        self.act2 = ReLU()
        self.pool2 = MaxPool2D(kernel_size=2, stride=2)
        self.head = Sequential(
            Linear(8 * 7 * 7, 32),
            ReLU(),
            Linear(32, 10),
        )

    def __call__(self, x: Tensor) -> Tensor:
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = x.flatten(start_dim=1)
        return self.head(x)

    def parameters(self):
        params = []
        for module in [self.conv1, self.conv2, self.head]:
            params.extend(module.parameters())
        return params


def main() -> None:
    np.random.seed(0)
    model = MiniConvNet()
    batch = Tensor(np.random.randn(8, 1, 28, 28))
    logits = model(batch)
    print("MNIST CNN smoke example")
    print("input shape:", batch.shape)
    print("logit shape:", logits.shape)
    print("parameters:", len(model.parameters()))
    print("This is a forward-pass smoke test. A real MNIST training loop is the next step.")


if __name__ == "__main__":
    main()
