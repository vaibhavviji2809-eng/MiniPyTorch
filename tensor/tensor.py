from __future__ import annotations

import math
from typing import Any

import numpy as np

from .autograd import backward as autograd_backward
from .ops import ensure_array, reduce_broadcasted_grad


class Tensor:
    _counter = 0

    def __init__(
        self,
        data: Any,
        requires_grad: bool = False,
        parents: tuple["Tensor", ...] = (),
        op: str = "",
        name: str | None = None,
    ) -> None:
        self.data = ensure_array(data)
        self.requires_grad = requires_grad
        self.grad: np.ndarray | None = np.zeros_like(self.data) if requires_grad else None
        self.parents = parents
        self.op = op
        self._backward = lambda: None
        if name is None:
            Tensor._counter += 1
            name = f"tensor_{Tensor._counter}"
        self.name = name

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def _coerce_grad(self, grad: Any) -> np.ndarray:
        grad_array = ensure_array(grad)
        if grad_array.shape == () and self.data.shape != ():
            grad_array = np.ones_like(self.data) * grad_array
        return grad_array

    def _add_grad(self, grad: np.ndarray) -> None:
        if not self.requires_grad:
            return
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad = self.grad + grad

    @staticmethod
    def ensure_tensor(value: Any) -> "Tensor":
        return value if isinstance(value, Tensor) else Tensor(value)

    def zero_grad(self) -> None:
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def detach(self) -> "Tensor":
        return Tensor(self.data.copy(), requires_grad=False)

    def item(self) -> float:
        return float(self.data.item())

    def numpy(self) -> np.ndarray:
        return self.data.copy()

    def __repr__(self) -> str:
        return (
            f"Tensor(data={self.data}, requires_grad={self.requires_grad}, "
            f"grad={self.grad})"
        )

    def __add__(self, other: Any) -> "Tensor":
        other = Tensor.ensure_tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            parents=(self, other),
            op="add",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self._add_grad(reduce_broadcasted_grad(out.grad, self.shape))
            if other.requires_grad:
                other._add_grad(reduce_broadcasted_grad(out.grad, other.shape))

        out._backward = _backward
        return out

    def __radd__(self, other: Any) -> "Tensor":
        return self + other

    def __neg__(self) -> "Tensor":
        out = Tensor(-self.data, requires_grad=self.requires_grad, parents=(self,), op="neg")

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self._add_grad(-out.grad)

        out._backward = _backward
        return out

    def __sub__(self, other: Any) -> "Tensor":
        return self + (-Tensor.ensure_tensor(other))

    def __rsub__(self, other: Any) -> "Tensor":
        return Tensor.ensure_tensor(other) - self

    def __mul__(self, other: Any) -> "Tensor":
        other = Tensor.ensure_tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            parents=(self, other),
            op="mul",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self._add_grad(
                    reduce_broadcasted_grad(out.grad * other.data, self.shape)
                )
            if other.requires_grad:
                other._add_grad(
                    reduce_broadcasted_grad(out.grad * self.data, other.shape)
                )

        out._backward = _backward
        return out

    def __rmul__(self, other: Any) -> "Tensor":
        return self * other

    def __truediv__(self, other: Any) -> "Tensor":
        other = Tensor.ensure_tensor(other)
        return self * other.pow(-1.0)

    def __rtruediv__(self, other: Any) -> "Tensor":
        return Tensor.ensure_tensor(other) / self

    def pow(self, power: float) -> "Tensor":
        out = Tensor(
            np.power(self.data, power),
            requires_grad=self.requires_grad,
            parents=(self,),
            op=f"pow({power})",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self._add_grad(out.grad * power * np.power(self.data, power - 1))

        out._backward = _backward
        return out

    def __pow__(self, power: float) -> "Tensor":
        return self.pow(power)

    def __matmul__(self, other: Any) -> "Tensor":
        other = Tensor.ensure_tensor(other)
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            parents=(self, other),
            op="matmul",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self._add_grad(out.grad @ other.data.T)
            if other.requires_grad:
                other._add_grad(self.data.T @ out.grad)

        out._backward = _backward
        return out

    @property
    def T(self) -> "Tensor":
        out = Tensor(
            self.data.T,
            requires_grad=self.requires_grad,
            parents=(self,),
            op="transpose",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self._add_grad(out.grad.T)

        out._backward = _backward
        return out

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            parents=(self,),
            op="sum",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            grad = out.grad
            if axis is None:
                grad = np.ones_like(self.data) * grad
            else:
                axes = (axis,) if isinstance(axis, int) else axis
                if not keepdims:
                    for ax in sorted(axes):
                        grad = np.expand_dims(grad, axis=ax)
                grad = np.ones_like(self.data) * grad
            self._add_grad(grad)

        out._backward = _backward
        return out

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        divisor = self.data.size if axis is None else np.prod(np.array(self.data.shape)[list((axis,) if isinstance(axis, int) else axis)])
        return self.sum(axis=axis, keepdims=keepdims) / divisor

    def exp(self) -> "Tensor":
        out = Tensor(
            np.exp(self.data),
            requires_grad=self.requires_grad,
            parents=(self,),
            op="exp",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self._add_grad(out.grad * out.data)

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        out = Tensor(
            np.log(self.data),
            requires_grad=self.requires_grad,
            parents=(self,),
            op="log",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self._add_grad(out.grad / self.data)

        out._backward = _backward
        return out

    def relu(self) -> "Tensor":
        out = Tensor(
            np.maximum(0.0, self.data),
            requires_grad=self.requires_grad,
            parents=(self,),
            op="relu",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self._add_grad(out.grad * (self.data > 0))

        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        sig = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(sig, requires_grad=self.requires_grad, parents=(self,), op="sigmoid")

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self._add_grad(out.grad * sig * (1 - sig))

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        tanh_value = np.tanh(self.data)
        out = Tensor(tanh_value, requires_grad=self.requires_grad, parents=(self,), op="tanh")

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self._add_grad(out.grad * (1 - tanh_value**2))

        out._backward = _backward
        return out

    def reshape(self, *shape: int) -> "Tensor":
        out = Tensor(
            self.data.reshape(*shape),
            requires_grad=self.requires_grad,
            parents=(self,),
            op="reshape",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self._add_grad(out.grad.reshape(self.shape))

        out._backward = _backward
        return out

    def flatten(self, start_dim: int = 1) -> "Tensor":
        if start_dim < 0:
            start_dim = self.ndim + start_dim
        prefix = self.shape[:start_dim]
        suffix = self.shape[start_dim:]
        flattened = int(np.prod(suffix)) if suffix else 1
        return self.reshape(*prefix, flattened)

    def backward(self, grad: Any | None = None) -> None:
        autograd_backward(self, grad=grad)
