from __future__ import annotations
import random
from pathlib import Path
from collections import deque
import heapq
import math

def bfs_shortest_path(n: int, adj: list[list[int]], s: int, t: int) -> list[int] | None:
    if s == t:
        return [s]
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

def dijkstra_shortest_path(n: int, adjw: list[list[tuple[int,int]]], s: int, t: int) -> tuple[int, list[int]] | None:
    if s == t:
        return (0, [s])
    dist = [math.inf] * (n + 1)
    parent = [-1] * (n + 1)
    dist[s] = 0
    parent[s] = s
    pq = [(0, s)]
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        if u == t:
            break
        for v, w in adjw[u]:
            # assume w >= 0
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
    return (dist[t], path)

def make_backbone_nodes(n: int) -> list[int]:
    """Pick a moderate-length chain from 1 to n to guarantee reachability."""
    if n <= 2:
        return [1, n]
    # backbone length grows but capped to avoid too many forced edges
    k = min(max(10, n // 200), 5000)  # internal nodes count
    internal_count = min(k, n - 2)
    internal = random.sample(range(2, n), internal_count)
    # keep chain somewhat long: sort gives increasing order, but not required
    # for shortest path, chain length helps create non-trivial routes
    internal.sort()
    return [1] + internal + [n]

def gen_unweighted_edges(n: int, m: int, ensure_path: bool, impossible: bool, seed: int,
                         cycle_rate: float = 0.25) -> list[tuple[int,int]]:
    random.seed(seed)
    edges_set: set[tuple[int,int]] = set()
    edges: list[tuple[int,int]] = []

    def can_add(u: int, v: int) -> bool:
        if u == v:
            return False
        if impossible and v == n:
            return False  # no incoming to n => unreachable
        if (u, v) in edges_set:
            return False
        return True

    def add(u: int, v: int):
        if can_add(u, v):
            edges_set.add((u, v))
            edges.append((u, v))
            return True
        return False

    if ensure_path and not impossible:
        chain = make_backbone_nodes(n)
        for a, b in zip(chain, chain[1:]):
            add(a, b)

    # Fill until m
    while len(edges) < m:
        u = random.randint(1, n)
        if random.random() < cycle_rate:
            # allow backward edge to create cycles
            v = random.randint(1, n)
        else:
            v = random.randint(1, n)
        add(u, v)

    return edges[:m]

def gen_weighted_edges(n: int, m: int, ensure_path: bool, impossible: bool, seed: int,
                       w_range: tuple[int,int] = (0, 100),
                       parallel_prob: float = 0.12,
                       cycle_rate: float = 0.25) -> list[tuple[int,int,int]]:
    random.seed(seed)
    edges: list[tuple[int,int,int]] = []
    pairs_seen: list[tuple[int,int]] = []  # for creating parallel edges

    def can_add(u: int, v: int) -> bool:
        if u == v:
            return False
        if impossible and v == n:
            return False
        return True

    def add(u: int, v: int, w: int):
        if can_add(u, v):
            edges.append((u, v, w))
            pairs_seen.append((u, v))
            return True
        return False

    if ensure_path and not impossible:
        chain = make_backbone_nodes(n)
        for a, b in zip(chain, chain[1:]):
            w = random.randint(*w_range)
            add(a, b, w)

    while len(edges) < m:
        # with some probability create parallel edge using an existing pair
        if pairs_seen and random.random() < parallel_prob:
            u, v = random.choice(pairs_seen)
        else:
            u = random.randint(1, n)
            if random.random() < cycle_rate:
                v = random.randint(1, n)
            else:
                v = random.randint(1, n)
        w = random.randint(*w_range)  # w >= 0
        add(u, v, w)

    return edges[:m]

def build_adj_unweighted(n: int, edges: list[tuple[int,int]]) -> list[list[int]]:
    adj = [[] for _ in range(n + 1)]
    for u, v in edges:
        adj[u].append(v)
    return adj

def build_adj_weighted(n: int, edges: list[tuple[int,int,int]]) -> list[list[tuple[int,int]]]:
    adj = [[] for _ in range(n + 1)]
    for u, v, w in edges:
        adj[u].append((v, w))
    return adj

def build_plan_35() -> list[dict]:
    plan = []

    # 1-10 small (mix: ensure_path and impossible)
    small = [
        (50, 200, True,  False),
        (100, 400, True, False),
        (200, 900, True, False),
        (300, 1400, True, False),
        (500, 2500, True, False),
        (800, 4500, True, False),
        (1200, 7000, True, False),
        (1500, 8000, True, False),
        (2000, 12000, True, False),
        (2500, 9000, False, True),
    ]
    for n, m, ok, imp in small:
        plan.append({"n": n, "m": m, "ensure": ok, "imp": imp})

    # 11-25 medium
    med = [
        (5000, 30000, True, False),
        (7000, 45000, True, False),
        (10000, 65000, True, False),
        (12000, 80000, True, False),
        (15000, 95000, True, False),
        (18000, 110000, True, False),
        (22000, 125000, True, False),
        (26000, 135000, True, False),
        (30000, 145000, True, False),
        (34000, 155000, True, False),
        (38000, 165000, True, False),
        (40000, 170000, True, False),
        (25000, 90000, False, True),
        (32000, 150000, True, False),
        (40000, 175000, True, False),
    ]
    for n, m, ok, imp in med:
        plan.append({"n": n, "m": m, "ensure": ok, "imp": imp})

    # 26-30 large
    large = [
        (60000, 180000, True, False),
        (70000, 185000, True, False),
        (80000, 190000, True, False),
        (90000, 195000, True, False),
        (90000, 195000, True, False),
    ]
    for n, m, ok, imp in large:
        plan.append({"n": n, "m": m, "ensure": ok, "imp": imp})

    # 31-35 stress (target 1-2s)
    for _ in range(5):
        plan.append({"n": 100000, "m": 200000, "ensure": True, "imp": False})

    assert len(plan) == 35
    return plan

def write_unweighted_suite(out_dir: str, base_seed: int = 111, cycle_rate: float = 0.25):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan = build_plan_35()

    for idx, spec in enumerate(plan, start=1):
        n, m, ensure, imp = spec["n"], spec["m"], spec["ensure"], spec["imp"]
        edges = gen_unweighted_edges(n, m, ensure_path=ensure, impossible=imp, seed=base_seed + idx, cycle_rate=cycle_rate)
        adj = build_adj_unweighted(n, edges)
        path = bfs_shortest_path(n, adj, 1, n)

        inp = out / f"{idx}.txt"
        ans = out / f"{idx}.ans"

        with inp.open("w", encoding="utf-8") as f:
            f.write(f"{n} {len(edges)}\n")
            f.write(f"1 {n}\n")
            for u, v in edges:
                f.write(f"{u} {v}\n")

        with ans.open("w", encoding="utf-8") as f:
            if path is None:
                f.write("-1\n")
            else:
                f.write(f"{len(path)}\n")
                f.write(" ".join(map(str, path)) + "\n")

def write_weighted_suite(out_dir: str, base_seed: int = 777, w_range: tuple[int,int] = (0, 100),
                         parallel_prob: float = 0.12, cycle_rate: float = 0.25):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan = build_plan_35()

    for idx, spec in enumerate(plan, start=1):
        n, m, ensure, imp = spec["n"], spec["m"], spec["ensure"], spec["imp"]
        edges = gen_weighted_edges(
            n, m, ensure_path=ensure, impossible=imp, seed=base_seed + idx,
            w_range=w_range, parallel_prob=parallel_prob, cycle_rate=cycle_rate
        )
        adjw = build_adj_weighted(n, edges)
        sol = dijkstra_shortest_path(n, adjw, 1, n)

        inp = out / f"{idx}.txt"
        ans = out / f"{idx}.ans"

        with inp.open("w", encoding="utf-8") as f:
            f.write(f"{n} {len(edges)}\n")
            f.write(f"1 {n}\n")
            for u, v, w in edges:
                f.write(f"{u} {v} {w}\n")

        with ans.open("w", encoding="utf-8") as f:
            if sol is None:
                f.write("-1\n")
            else:
                cost, path = sol
                f.write(f"{int(cost)}\n")
                f.write(" ".join(map(str, path)) + "\n")

def main():
    write_unweighted_suite("tests_shortest_unweighted", base_seed=111, cycle_rate=0.30)
    write_weighted_suite("tests_shortest_weighted", base_seed=777, w_range=(0, 200),
                         parallel_prob=0.15, cycle_rate=0.30)
    print("Done. Generated 35 tests for unweighted + 35 tests for weighted.")

if __name__ == "__main__":
    main()
