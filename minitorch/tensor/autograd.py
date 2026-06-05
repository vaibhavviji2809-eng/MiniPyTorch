from __future__ import annotations

from collections.abc import Iterable


def topological_sort(root) -> list:
    order: list = []
    visited: set[int] = set()

    def visit(node) -> None:
        if id(node) in visited:
            return
        visited.add(id(node))
        for parent in node.parents:
            visit(parent)
        order.append(node)

    visit(root)
    return order


def backward(root, grad=None) -> None:
    if grad is None:
        grad = 1.0

    root.grad = root._coerce_grad(grad)
    for node in reversed(topological_sort(root)):
        node._backward()
