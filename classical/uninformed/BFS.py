from typing import Callable, List, Optional, Tuple, Dict, Any
from collections import deque

def BFS(adj: List[List[int]], s: int, t: int) -> Optional[List[int]]:
    if s == t:
        return [s]
    n = len(adj) - 1
    parent = [-1] * (n + 1)
    parent[s] = s
    q = deque([s])

    while q:
        u = q.popleft()
        for v in adj[u]:
            if parent[v] == -1:
                parent[v] = u
                if v == t:
                    q.clear()
                    break
                q.append(v)

    if parent[t] == -1:
        return None
    path = []
    cur = t
    while cur != parent[cur]:
        path.append(cur)
        cur = parent[cur]
    path.append(s)
    path.reverse()
    return path