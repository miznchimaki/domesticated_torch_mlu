"""
@Copyright (C) [2022-2025] by Cambricon.
@File    :   utils.py
"""
from torch.fx.node import Node
from collections import deque


def topo_sort_nodes(nodes: list[Node]) -> list[Node]:
    """
    Perform a topological sort on the given list of nodes based on dependencies
    """
    nodes_set = set(nodes)
    indegree = {node: 0 for node in nodes}
    adj = {node: [] for node in nodes}

    for node in nodes:
        deps = []

        def extract_deps(arg):
            if isinstance(arg, Node) and arg in nodes_set:
                deps.append(arg)
            elif isinstance(arg, (list, tuple)):
                for a in arg:
                    extract_deps(a)
            elif isinstance(arg, dict):
                for v in arg.values():
                    extract_deps(v)

        extract_deps(node.args)
        extract_deps(node.kwargs)

        for dep in deps:
            adj[dep].append(node)
            indegree[node] += 1

    queue = deque([n for n, deg in indegree.items() if deg == 0])
    sorted_nodes = []

    while queue:
        n = queue.popleft()
        sorted_nodes.append(n)
        for nei in adj[n]:
            indegree[nei] -= 1
            if indegree[nei] == 0:
                queue.append(nei)

    if len(sorted_nodes) != len(nodes):
        raise RuntimeError("Graph has a cycle or missing dependencies")

    return sorted_nodes
