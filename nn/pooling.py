from __future__ import annotations

import numpy as np

from tensor.tensor import Tensor
from .module import Module


class MaxPool2D(Module):
    def __init__(self, kernel_size: int | tuple[int, int], stride: int | None = None) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size[0]

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError("MaxPool2D expects input with shape (batch, channels, height, width).")

        batch_size, channels, height, width = x.shape
        kh, kw = self.kernel_size
        out_height = ((height - kh) // self.stride) + 1
        out_width = ((width - kw) // self.stride) + 1

        output = np.zeros((batch_size, channels, out_height, out_width), dtype=np.float64)
        max_indices: list[tuple[int, int, int, int, int, int]] = []

        for batch in range(batch_size):
            for channel in range(channels):
                for out_row in range(out_height):
                    row_start = out_row * self.stride
                    for out_col in range(out_width):
                        col_start = out_col * self.stride
                        region = x.data[
                            batch,
                            channel,
                            row_start : row_start + kh,
                            col_start : col_start + kw,
                        ]
                        max_index = int(np.argmax(region))
                        max_row, max_col = np.unravel_index(max_index, region.shape)
                        output[batch, channel, out_row, out_col] = region[max_row, max_col]
                        max_indices.append(
                            (batch, channel, out_row, out_col, row_start + max_row, col_start + max_col)
                        )

        out = Tensor(
            output,
            requires_grad=x.requires_grad,
            parents=(x,),
            op="maxpool2d",
        )

        def _backward() -> None:
            if out.grad is None or not x.requires_grad:
                return
            grad_input = np.zeros_like(x.data)
            for batch, channel, out_row, out_col, max_row, max_col in max_indices:
                grad_input[batch, channel, max_row, max_col] += out.grad[batch, channel, out_row, out_col]
            x._add_grad(grad_input)

        out._backward = _backward
        return out
