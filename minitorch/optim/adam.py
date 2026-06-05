from __future__ import annotations

import numpy as np


class Adam:
    def __init__(
        self,
        params,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(param.data) for param in self.params]
        self.v = [np.zeros_like(param.data) for param in self.params]

    def zero_grad(self) -> None:
        for param in self.params:
            param.zero_grad()

    def step(self) -> None:
        self.t += 1
        for index, param in enumerate(self.params):
            if param.grad is None:
                continue
            self.m[index] = self.beta1 * self.m[index] + (1 - self.beta1) * param.grad
            self.v[index] = self.beta2 * self.v[index] + (1 - self.beta2) * (param.grad**2)

            m_hat = self.m[index] / (1 - self.beta1**self.t)
            v_hat = self.v[index] / (1 - self.beta2**self.t)
            param.data = param.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
