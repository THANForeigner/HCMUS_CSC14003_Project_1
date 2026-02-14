from typing import Callable, List, Optional, Tuple, Dict, Any
import heapq

def _zero_heuristic(_: int, __: int) -> int:
    return 0

def GBFS(
    adjw: List[List[Tuple[int, int]]],
    s: int,
    t: int,
    h: Callable[[int, int], int] = _zero_heuristic
) -> Optional[Tuple[int, List[int]]]:
    if s == t:
        return (0, [s])
    n = len(adjw) - 1
    visited = [False] * (n + 1)
    parent = [-1] * (n + 1)

    pq = [(h(s, t), s)]
    parent[s] = s

    while pq:
        _, u = heapq.heappop(pq)
        if visited[u]:
            continue
        visited[u] = True
        if u == t:
            break
        for v, w in adjw[u]:
            if not visited[v]:
                if parent[v] == -1:
                    parent[v] = u
                heapq.heappush(pq, (h(v, t), v))

    if not visited[t]:
        return None

    # reconstruct
    path = []
    cur = t
    while cur != parent[cur]:
        path.append(cur)
        cur = parent[cur]
    path.append(s)
    path.reverse()

    # compute cost along the path (pick first matching edge a->b)
    cost = 0
    for a, b in zip(path, path[1:]):
        found = False
        for v, w in adjw[a]:
            if v == b:
                cost += w
                found = True
                break
        if not found:
            # should not happen if parent edges are consistent
            return None

    return (cost, path)
