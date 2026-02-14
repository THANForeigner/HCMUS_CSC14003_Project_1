from typing import Callable, List, Optional, Tuple, Dict, Any
import heapq
import math

def dijkstra(adjw: List[List[Tuple[int, int]]], s: int, t: int) -> Optional[Tuple[int, List[int]]]:
    """Shortest path by total weight (w >= 0). Supports multi-edge naturally."""
    if s == t:
        return (0, [s])
    n = len(adjw) - 1
    dist = [math.inf] * (n + 1)
    parent = [-1] * (n + 1)

    dist[s] = 0
    parent[s] = s
    pq = [(0, s)]  # (dist, node)

    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        if u == t:
            break
        for v, w in adjw[u]:
            if w < 0:
                raise ValueError("Dijkstra/UCS requires non-negative weights.")
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))

    if parent[t] == -1:
        return None

    path = []
    cur = t
    while cur != parent[cur]:
        path.append(cur)
        cur = parent[cur]
    path.append(s)
    path.reverse()
    return (int(dist[t]), path)


def UCS(adjw: List[List[Tuple[int, int]]], s: int, t: int) -> Optional[Tuple[int, List[int]]]:
    """Uniform Cost Search = Dijkstra (với w >= 0)."""
    return dijkstra(adjw, s, t)