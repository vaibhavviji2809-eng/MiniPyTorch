# MiniPyTorch

A from-scratch deep learning framework that implements tensors, autograd, layers, losses, and optimizers without importing PyTorch.

Final target usage:

```python
from tensor.tensor import Tensor

x = Tensor([[1, 2], [3, 4]], requires_grad=True)
w = Tensor([[0.5], [0.8]], requires_grad=True)
y = x @ w
loss = y.mean()
loss.backward()
print(w.grad)
```

## Repository Structure

```text
MiniPyTorch/
|- tensor/
|  |- tensor.py
|  |- ops.py
|  |- autograd.py
|  `- graph.py
|- nn/
|  |- module.py
|  |- linear.py
|  |- embedding.py
|  |- norm.py
|  |- transformer.py
|  |- activations.py
|  `- losses.py
|- optim/
|  |- sgd.py
|  `- adam.py
|- datasets/
|  |- dataset.py
|  `- char.py
|- examples/
|  |- linear_regression.py
|  |- xor.py
|  |- mnist.py
|  `- tiny_gpt.py
`- tests/
```

## Implemented

- tensor class with `shape`, `requires_grad`, and `grad`
- arithmetic ops: `+`, `-`, `*`, `/`, `@`
- computation graph via parent tracking
- reverse-mode autograd with topological sorting
- activations: `ReLU`, `Sigmoid`, `Tanh`
- module system with `Module`, `Linear`, and `Sequential`
- vision layers with `Conv2D` and `MaxPool2D`
- dataset and dataloader utilities
- `Embedding`, `LayerNorm`, `MultiHeadAttention`, and `TransformerBlock`
- losses: `MSE`, binary cross entropy, multiclass cross entropy
- optimizers: `SGD`, `Adam`
- examples: linear regression and XOR
- CNN smoke example for future MNIST work
- Tiny GPT training example

## Not Finished Yet

- a real MNIST training example
- GPU backend

## Run Examples

```bash
py examples/linear_regression.py
py examples/xor.py
```

## Run Tests

```bash
py -m unittest discover tests
```
