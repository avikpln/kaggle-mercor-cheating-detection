from collections import deque
import networkx as nx


def directed_reachable(graph, sources):
    reached = set(sources) & set(graph.nodes)
    queue = deque(reached)
    while queue:
        u = queue.popleft()
        for v in graph.successors(u):
            if v not in reached:
                reached.add(v)
                queue.append(v)
    return reached
