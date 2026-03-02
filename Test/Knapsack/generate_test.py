from __future__ import annotations
import random
from pathlib import Path

OUTPUT_DIR = "tests_knapsack"

def solve_knapsack_dp(n, capacity, weights, values):
    K = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

    for i in range(n + 1):
        for w in range(capacity + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif weights[i - 1] <= w:
                val_take = values[i - 1] + K[i - 1][w - weights[i - 1]]
                val_skip = K[i - 1][w]
                K[i][w] = max(val_take, val_skip)
            else:
                K[i][w] = K[i - 1][w]
    max_val = K[n][capacity]
    res = max_val
    w = capacity
    selection = [0] * n

    for i in range(n, 0, -1):
        if res <= 0:
            break
        if res != K[i - 1][w]:
            selection[i - 1] = 1
            res -= values[i - 1]
            w -= weights[i - 1]

    return max_val, selection


def gen_knapsack_data(n: int, seed: int, max_weight=50, max_value=100):
    random.seed(seed)
    weights = []
    values = []
    for _ in range(n):
        w = random.randint(1, max_weight)
        v = random.randint(1, max_value)
        weights.append(w)
        values.append(v)
    total_weight = sum(weights)
    capacity = int(total_weight * 0.5)
    return capacity, weights, values


def build_plan_kp():
    plan = []
    plan.append({"n": 250})
    plan.append({"n": 300})
    plan.append({"n": 350})
    plan.append({"n": 400})
    plan.append({"n": 450})
    plan.append({"n": 500})

    plan.append({"n": 600})
    plan.append({"n": 700})
    plan.append({"n": 800})
    plan.append({"n": 900})
    plan.append({"n": 1000})

    plan.append({"n": 1200})
    plan.append({"n": 1500})
    plan.append({"n": 1800})
    plan.append({"n": 2000})

    return plan

def write_knapsack_suite(base_seed: int = 2024):
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    plan = build_plan_kp()

    print(f"Generating {len(plan)} tests for Knapsack Problem...")

    for idx, spec in enumerate(plan, start=1):
        n = spec["n"]

        capacity, weights, values = gen_knapsack_data(n, seed=base_seed + idx)

        max_val, selection = solve_knapsack_dp(n, capacity, weights, values)

        inp_path = out / f"{idx}.txt"
        with inp_path.open("w", encoding="utf-8") as f:
            f.write(f"{n} {capacity}\n")
            for w, v in zip(weights, values):
                f.write(f"{w} {v}\n")

        ans_path = out / f"{idx}.ans"
        with ans_path.open("w", encoding="utf-8") as f:
            f.write(f"{max_val}\n")
            f.write(" ".join(map(str, selection)) + "\n")

        print(f"Generated test {idx}: N={n}, Capacity={capacity}, MaxVal={max_val}")


def main():
    write_knapsack_suite()
    print(f"Done. Files saved to '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()