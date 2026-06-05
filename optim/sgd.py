from __future__ import annotations


class SGD:
    def __init__(self, params, lr: float = 0.01) -> None:
        self.params = list(params)
        self.lr = lr

    def zero_grad(self) -> None:
        for param in self.params:
            param.zero_grad()

    def step(self) -> None:
        for param in self.params:
            if param.grad is None:
                continue
            param.data = param.data - self.lr * param.grad
