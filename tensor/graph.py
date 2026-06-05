from __future__ import annotations


def trace(root) -> list[tuple[str, str, str]]:
    edges: list[tuple[str, str, str]] = []
    visited: set[int] = set()

    def walk(node) -> None:
        if id(node) in visited:
            return
        visited.add(id(node))
        for parent in node.parents:
            edges.append((parent.name, node.name, node.op))
            walk(parent)

    walk(root)
    return edges
