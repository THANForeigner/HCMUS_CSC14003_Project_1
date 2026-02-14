import random
import math
from pathlib import Path
import itertools

OUTPUT_DIR = "tests_tsp"


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def solve_tsp_dynamic_programming(n, points):
    """
    Giải chính xác TSP bằng quy hoạch động (Held-Karp).
    Chỉ dùng cho N <= 16. Độ phức tạp O(n^2 * 2^n).
    """
    dist = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist[i][j] = distance(points[i], points[j])
    memo = {}

    for i in range(1, n):
        memo[(1 | (1 << i), i)] = (dist[0][i], 0)

    for r in range(3, n + 1):
        for subset in itertools.combinations(range(1, n), r - 1):
            mask = 1
            for node in subset:
                mask |= (1 << node)

            for next_node in subset:
                prev_mask = mask ^ (1 << next_node)
                min_val = float('inf')
                parent = -1

                for prev_node in range(n):
                    if prev_node != next_node and (prev_mask & (1 << prev_node)):
                        cost, _ = memo.get((prev_mask, prev_node), (float('inf'), -1))
                        if cost + dist[prev_node][next_node] < min_val:
                            min_val = cost + dist[prev_node][next_node]
                            parent = prev_node

                if parent != -1:
                    memo[(mask, next_node)] = (min_val, parent)

    full_mask = (1 << n) - 1
    min_tour_cost = float('inf')
    last_node = -1

    for i in range(1, n):
        cost, _ = memo.get((full_mask, i), (float('inf'), -1))
        total_cost = cost + dist[i][0]
        if total_cost < min_tour_cost:
            min_tour_cost = total_cost
            last_node = i

    # Truy vết đường đi
    path = [0]
    curr = last_node
    curr_mask = full_mask
    temp_path = []

    while curr != 0:
        temp_path.append(curr)
        _, parent = memo[(curr_mask, curr)]
        curr_mask = curr_mask ^ (1 << curr)
        curr = parent

    path.extend(reversed(temp_path))
    path.append(0)  # Quay về 0

    return min_tour_cost, path


def gen_tsp_test(n, width=100, height=100, seed=42):
    random.seed(seed)
    points = []
    # Sinh tọa độ ngẫu nhiên
    for _ in range(n):
        x = round(random.uniform(0, width), 2)
        y = round(random.uniform(0, height), 2)
        points.append((x, y))
    return points


def build_plan_tsp():
    plan = []

    plan.append({"n": 5})
    plan.append({"n": 8})
    plan.append({"n": 10})
    plan.append({"n": 12})
    plan.append({"n": 13})

    plan.append({"n": 16})
    plan.append({"n": 20})
    plan.append({"n": 25})
    plan.append({"n": 30})
    plan.append({"n": 40})

    plan.append({"n": 50})
    plan.append({"n": 60})
    plan.append({"n": 75})
    plan.append({"n": 100})
    plan.append({"n": 150})
    plan.append({"n": 200})

    return plan


def main():
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    plan = build_plan_tsp()

    print(f"Generating {len(plan)} TSP tests...")

    for idx, spec in enumerate(plan, start=1):
        n = spec["n"]
        points = gen_tsp_test(n, seed=2024 + idx)

        inp_path = out / f"{idx}.txt"
        with inp_path.open("w", encoding="utf-8") as f:
            # Format:
            # N
            # X Y ...
            f.write(f"{n}\n")
            for x, y in points:
                f.write(f"{x} {y}\n")

        ans_path = out / f"{idx}.ans"

        # Chỉ giải chính xác nếu N <= 16 (Vì thuật toán O(2^n) rất nặng)
        if n <= 16:
            print(f"Solving exact solution for Test {idx} (N={n})...")
            cost, path = solve_tsp_dynamic_programming(n, points)

            with ans_path.open("w", encoding="utf-8") as f:
                f.write(f"{cost:.4f}\n")
                # Path format: 0 1 3 2 0 (index 0-based)
                f.write(" ".join(map(str, path)) + "\n")
        else:
            # Với N lớn, không có đáp án chính xác tham chiếu
            # Ghi -1 để báo hiệu
            with ans_path.open("w", encoding="utf-8") as f:
                f.write("-1\n")

    print(f"Done! Check folder '{OUTPUT_DIR}'")


if __name__ == "__main__":
    main()