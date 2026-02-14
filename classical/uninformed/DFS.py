from typing import Callable, List, Optional, Tuple, Dict, Any

def DFS(adj: List[List[int]], s: int, t: int) -> Optional[List[int]]:
    if s == t:
        return [s]
    n = len(adj) - 1
    visited = [False] * (n + 1)
    parent = [-1] * (n + 1)

    stack = [s]
    visited[s] = True
    parent[s] = s

    while stack:
        u = stack.pop()
        if u == t:
            break
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                stack.append(v)

    if not visited[t]:
        return None

    path = []
    cur = t
    while cur != parent[cur]:
        path.append(cur)
        cur = parent[cur]
    path.append(s)
    path.reverse()
    return path