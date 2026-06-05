# MiniTorch

A small from-scratch deep learning framework that supports:

- tensor arithmetic
- computation graphs
- reverse-mode autodiff
- modules and layers
- optimizers
- runnable training examples

Target usage:

```python
from tensor.tensor import Tensor

x = Tensor([[1, 2], [3, 4]], requires_grad=True)
w = Tensor([[0.5], [0.8]], requires_grad=True)
y = x @ w
loss = y.mean()
loss.backward()
print(w.grad)
```

## Project Layout

```text
minitorch/
|- tensor/
|  |- tensor.py
|  |- ops.py
|  |- autograd.py
|  `- graph.py
|- nn/
|  |- module.py
|  |- linear.py
|  |- activations.py
|  `- losses.py
|- optim/
|  |- sgd.py
|  `- adam.py
|- examples/
|  |- linear_regression.py
|  |- xor.py
|  `- mnist.py
`- tests/
```

## Implemented Now

- tensor wrapper with `shape`, `requires_grad`, and `grad`
- scalar, elementwise, reduction, and matrix operations
- computation graph construction through tensor parents
- reverse-mode autodiff with topological sorting
- `relu`, `sigmoid`, and `tanh`
- `Module`, `Linear`, and `Sequential`
- `MSE`, binary cross entropy, and multiclass cross entropy
- `SGD` and `Adam`
- linear regression and XOR training examples

## Not Finished Yet

- a real MNIST training example
- convolution and pooling layers
- GPU backend

## Run examples

```bash
py minitorch/examples/linear_regression.py
py minitorch/examples/xor.py
```

## Run tests

```bash
py -m unittest discover minitorch/tests
```
