from __future__ import annotations

import math
import numpy as np

from tensor.tensor import Tensor
from .module import Module


class Conv2D(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        kh, kw = kernel_size
        bound = math.sqrt(2.0 / (in_channels * kh * kw))
        self.weight = Tensor(
            np.random.randn(out_channels, in_channels, kh, kw) * bound,
            requires_grad=True,
            name="conv_weight",
        )
        self.bias = Tensor(
            np.zeros((out_channels,)),
            requires_grad=True,
            name="conv_bias",
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError("Conv2D expects input with shape (batch, channels, height, width).")

        batch_size, in_channels, height, width = x.shape
        if in_channels != self.in_channels:
            raise ValueError("Input channel count does not match Conv2D configuration.")

        kh, kw = self.kernel_size
        padded = np.pad(
            x.data,
            ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
            mode="constant",
        )
        out_height = ((height + 2 * self.padding - kh) // self.stride) + 1
        out_width = ((width + 2 * self.padding - kw) // self.stride) + 1
        output = np.zeros((batch_size, self.out_channels, out_height, out_width), dtype=np.float64)

        for batch in range(batch_size):
            for out_channel in range(self.out_channels):
                for out_row in range(out_height):
                    row_start = out_row * self.stride
                    for out_col in range(out_width):
                        col_start = out_col * self.stride
                        region = padded[
                            batch,
                            :,
                            row_start : row_start + kh,
                            col_start : col_start + kw,
                        ]
                        output[batch, out_channel, out_row, out_col] = (
                            np.sum(region * self.weight.data[out_channel]) + self.bias.data[out_channel]
                        )

        out = Tensor(
            output,
            requires_grad=x.requires_grad or self.weight.requires_grad or self.bias.requires_grad,
            parents=(x, self.weight, self.bias),
            op="conv2d",
        )

        def _backward() -> None:
            if out.grad is None:
                return

            grad_input_padded = np.zeros_like(padded) if x.requires_grad else None
            grad_weight = np.zeros_like(self.weight.data) if self.weight.requires_grad else None
            grad_bias = np.zeros_like(self.bias.data) if self.bias.requires_grad else None

            for batch in range(batch_size):
                for out_channel in range(self.out_channels):
                    for out_row in range(out_height):
                        row_start = out_row * self.stride
                        for out_col in range(out_width):
                            col_start = out_col * self.stride
                            grad_value = out.grad[batch, out_channel, out_row, out_col]
                            region = padded[
                                batch,
                                :,
                                row_start : row_start + kh,
                                col_start : col_start + kw,
                            ]
                            if grad_weight is not None:
                                grad_weight[out_channel] += grad_value * region
                            if grad_input_padded is not None:
                                grad_input_padded[
                                    batch,
                                    :,
                                    row_start : row_start + kh,
                                    col_start : col_start + kw,
                                ] += grad_value * self.weight.data[out_channel]
                            if grad_bias is not None:
                                grad_bias[out_channel] += grad_value

            if x.requires_grad and grad_input_padded is not None:
                if self.padding > 0:
                    grad_input = grad_input_padded[
                        :,
                        :,
                        self.padding : self.padding + height,
                        self.padding : self.padding + width,
                    ]
                else:
                    grad_input = grad_input_padded
                x._add_grad(grad_input)
            if self.weight.requires_grad and grad_weight is not None:
                self.weight._add_grad(grad_weight)
            if self.bias.requires_grad and grad_bias is not None:
                self.bias._add_grad(grad_bias)

        out._backward = _backward
        return out
