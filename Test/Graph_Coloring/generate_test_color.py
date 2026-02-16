from __future__ import annotations
import random
from pathlib import Path

OUTPUT_DIR = "tests_graph_coloring"


def gen_k_colorable_graph(n: int, m: int, k: int, seed: int) -> tuple[list[tuple[int, int]], list[int]]:
    random.seed(seed)
    node_colors = [0] * (n + 1)
    nodes = list(range(1, n + 1))
    random.shuffle(nodes)

    for i in range(k):
        if i < n:
            node_colors[nodes[i]] = i + 1

    for i in range(k, n):
        node_colors[nodes[i]] = random.randint(1, k)

    edges_set: set[tuple[int, int]] = set()
    edges: list[tuple[int, int]] = []

    attempts = 0
    max_attempts = m * 200

    while len(edges) < m and attempts < max_attempts:
        attempts += 1
        u = random.randint(1, n)
        v = random.randint(1, n)
        if u == v: continue
        if node_colors[u] == node_colors[v]:
            continue
        if u > v: u, v = v, u
        if (u, v) in edges_set:
            continue
        edges_set.add((u, v))
        edges.append((u, v))
    return edges, node_colors

def build_plan_gc() -> list[dict]:
    plan = []
    plan.append({"n": 10, "m": 20, "k": 3})
    plan.append({"n": 15, "m": 35, "k": 3})
    plan.append({"n": 20, "m": 40, "k": 4})
    plan.append({"n": 20, "m": 60, "k": 3})
    plan.append({"n": 25, "m": 50, "k": 4})

    medium_configs = [
        (30, 70, 4),  # Test 6
        (35, 90, 5),
        (40, 100, 5),
        (45, 120, 6),
        (50, 150, 5),  # Test 10: N=50
        (55, 160, 7),
        (60, 180, 6),
        (65, 200, 8),
        (70, 220, 5),
        (75, 250, 10),  # Test 15
    ]
    for n, m, k in medium_configs:
        plan.append({"n": n, "m": m, "k": k})

    hard_configs = [
        (80, 280, 8),  # Test 16
        (85, 300, 12),
        (90, 320, 15),
        (95, 350, 20),
        (100, 400, 10),  # Test 20: N=100
        (105, 420, 15),
        (110, 450, 18),
        (115, 480, 20),
        (120, 500, 25),  #
        (130, 550, 30)  # Test 25: Max Ping.
    ]
    for n, m, k in hard_configs:
        plan.append({"n": n, "m": m, "k": k})
    return plan


def write_gc_suite(base_seed: int = 2024):
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    plan = build_plan_gc()

    print(f"Generating {len(plan)} tests for Graph Coloring...")
    print(f"Target: DFS & ACO both runnable. Max time approx 5s for last tests.")

    for idx, spec in enumerate(plan, start=1):
        n, m, k = spec["n"], spec["m"], spec["k"]
        edges, node_colors = gen_k_colorable_graph(n, m, k, seed=base_seed + idx)
        actual_m = len(edges)
        inp_path = out / f"{idx}.txt"

        with inp_path.open("w", encoding="utf-8") as f:
            f.write(f"{n} {actual_m}\n")
            for u, v in edges:
                f.write(f"{u} {v}\n")

        ans_path = out / f"{idx}.ans"
        with ans_path.open("w", encoding="utf-8") as f:
            f.write(f"{k}\n")
            colors_str = " ".join(map(str, node_colors[1:]))
            f.write(colors_str + "\n")

        if idx % 5 == 0:
            print(f"Generated test {idx}/{len(plan)} (N={n}, M={actual_m}, K={k})")


def main():
    write_gc_suite()
    print(f"Done. Files saved to '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()