"""
problems/input_validator.py — Input Data Hub
=============================================
Nơi duy nhất quản lý:
  1. PATH CONSTANTS — vị trí tất cả test dirs
  2. load_*_cases() — đọc & parse file test → list các case
  3. Continuous Optimization config — func names, dims, runs

Cách dùng trong benchmark:
    from problems.input_validator import load_knapsack_cases
    for case in load_knapsack_cases():
        run_solver(case.capacity, case.weights, case.values)

    from problems.input_validator import CONT_FUNCS, CONT_DIMS
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any, List, NamedTuple, Optional, Tuple

# Root path
# input_validator.py nằm ở problems/ → root là parent của problems/
_ROOT = Path(__file__).parent.parent

#Path constants 
KNAPSACK_DIR       = str(_ROOT / "Test" / "Knapsack"           / "tests_knapsack")
GRAPH_DIR          = str(_ROOT / "Test" / "Graph_Coloring"     / "tests_graph_coloring")
TSP_DIR            = str(_ROOT / "Test" / "Traveling_Sale_Man" / "tests_tsp")
SP_UNWEIGHTED_DIR  = str(_ROOT / "Test" / "Shortest_Path"      / "tests_shortest_unweighted")
SP_WEIGHTED_DIR    = str(_ROOT / "Test" / "Shortest_Path"      / "tests_shortest_weighted")

# Continuous Optimization config
CONT_FUNCS = ["sphere", "rastrigin", "rosenbrock", "griewank", "ackley"]
CONT_DIMS  = [10, 30]          # dimensions thường dùng để scalability test
CONT_RUNS  = 10                # số lần chạy mỗi thuật toán để lấy mean/std
CONT_ALGS  = ["HC", "SA", "DE", "PSO", "ABC", "FA", "CS", "TLBO"]

# Case types (NamedTuple)

class KnapsackCase(NamedTuple):
    id: int
    capacity: int
    weights: List[int]
    values: List[int]
    expected: Optional[float]

class GraphCase(NamedTuple):
    id: int
    n: int
    edges: List[Tuple[int, int]]
    expected_colors: Optional[int]

class TSPCase(NamedTuple):
    id: int
    points: List[Tuple[float, float]]
    expected_cost: Optional[float]

class SPCase(NamedTuple):
    id: int
    n: int
    s: int
    t: int
    edges: list          # unweighted: [(u,v)] | weighted: [(u,v,w)]
    expected: Any        # (cost, path) or None

class ContCase(NamedTuple):
    func_name: str       # e.g. "sphere"
    dim: int             # number of dimensions
    # func, lb, ub được lấy runtime qua get_problem(func_name) từ problems.problem

# Loaders 

def load_knapsack_cases(tests_dir: str = KNAPSACK_DIR, num: int = 15) -> List[KnapsackCase]:
    """Đọc tất cả test cases Knapsack. Format file: line1='n cap', các dòng sau='w v'"""
    cases = []
    for i in range(1, num + 1):
        inp = os.path.join(tests_dir, f"{i}.txt")
        if not os.path.exists(inp):
            continue
        try:
            with open(inp, encoding="utf-8") as f:
                n, cap = map(int, f.readline().split())
                weights, values = [], []
                for _ in range(n):
                    parts = list(map(int, f.readline().split()))
                    if len(parts) >= 2:
                        weights.append(parts[0]); values.append(parts[1])
        except Exception:
            continue
        expected = _read_ans_float(os.path.join(tests_dir, f"{i}.ans"))
        cases.append(KnapsackCase(i, cap, weights, values, expected))
    return cases


def load_graph_cases(tests_dir: str = GRAPH_DIR, num: int = 12) -> List[GraphCase]:
    """Đọc tất cả test cases Graph Coloring. Format: line1='n m', các dòng sau='u v'"""
    cases = []
    for i in range(1, num + 1):
        inp = os.path.join(tests_dir, f"{i}.txt")
        if not os.path.exists(inp):
            continue
        try:
            with open(inp, encoding="utf-8") as f:
                n, m = map(int, f.readline().split())
                edges = []
                for _ in range(m):
                    parts = list(map(int, f.readline().split()))
                    if len(parts) >= 2:
                        edges.append((parts[0], parts[1]))
        except Exception:
            continue
        expected = _read_ans_int(os.path.join(tests_dir, f"{i}.ans"))
        cases.append(GraphCase(i, n, edges, expected))
    return cases


def load_tsp_cases(tests_dir: str = TSP_DIR, num: int = 16) -> List[TSPCase]:
    """Đọc tất cả test cases TSP. Format: line1='n', các dòng sau='x y'"""
    cases = []
    for i in range(1, num + 1):
        inp = os.path.join(tests_dir, f"{i}.txt")
        if not os.path.exists(inp):
            continue
        try:
            with open(inp, encoding="utf-8") as f:
                n = int(f.readline().strip())
                points = []
                for _ in range(n):
                    parts = list(map(float, f.readline().split()))
                    if len(parts) >= 2:
                        points.append((parts[0], parts[1]))
        except Exception:
            continue
        expected = _read_ans_float(os.path.join(tests_dir, f"{i}.ans"), neg1_is_none=True)
        cases.append(TSPCase(i, points, expected))
    return cases


def load_sp_unweighted_cases(tests_dir: str = SP_UNWEIGHTED_DIR, num: int = 12) -> List[SPCase]:
    """Đọc test cases Shortest Path (unweighted). Format: line1='n m', line2='s t', các dòng sau='u v'"""
    return _load_sp_cases(tests_dir, num, weighted=False)


def load_sp_weighted_cases(tests_dir: str = SP_WEIGHTED_DIR, num: int = 12) -> List[SPCase]:
    """Đọc test cases Shortest Path (weighted). Format: line1='n m', line2='s t', các dòng sau='u v w'"""
    return _load_sp_cases(tests_dir, num, weighted=True)


def load_cont_cases(funcs=None, dims=None) -> List[ContCase]:
    """
    Tạo danh sách tất cả (func_name, dim) cần benchmark cho Continuous Optimization.
    Không cần file — config nằm ngay trong CONT_FUNCS / CONT_DIMS.

    Cách dùng:
        from problems.input_validator import load_cont_cases
        from problems.problem import get_problem
        for case in load_cont_cases():
            prob = get_problem(case.func_name)
            run_solver(prob['func'], prob['lb'], prob['ub'], case.dim)
    """
    funcs = funcs or CONT_FUNCS
    dims  = dims  or CONT_DIMS
    return [ContCase(fn, d) for fn in funcs for d in dims]


# Internal helpers 

def _load_sp_cases(tests_dir: str, num: int, weighted: bool) -> List[SPCase]:
    cases = []
    for i in range(1, num + 1):
        inp = os.path.join(tests_dir, f"{i}.txt")
        if not os.path.exists(inp):
            continue
        try:
            with open(inp, encoding="utf-8") as f:
                n, m = map(int, f.readline().split())
                s, t = map(int, f.readline().split())
                edges = [tuple(map(int, f.readline().split())) for _ in range(m)]
        except Exception:
            continue
        expected = None
        ans = os.path.join(tests_dir, f"{i}.ans")
        if os.path.exists(ans):
            try:
                with open(ans, encoding="utf-8") as f:
                    first = f.readline().strip()
                    if first not in ("", "-1"):
                        cost = int(first)
                        path = list(map(int, f.readline().split()))
                        expected = (cost, path)
            except Exception:
                pass
        cases.append(SPCase(i, n, s, t, edges, expected))
    return cases


def _read_ans_float(path: str, neg1_is_none: bool = False) -> Optional[float]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            v = float(f.readline().strip())
            return None if (neg1_is_none and v == -1) else v
    except Exception:
        return None


def _read_ans_int(path: str) -> Optional[int]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            line = f.readline().strip()
            return int(line) if line else None
    except Exception:
        return None


# Self-test 

if __name__ == "__main__":
    print("[ Paths ]")
    for name, d in [("Knapsack", KNAPSACK_DIR), ("Graph", GRAPH_DIR),
                    ("TSP", TSP_DIR), ("SP-U", SP_UNWEIGHTED_DIR), ("SP-W", SP_WEIGHTED_DIR)]:
        ok = "✅" if os.path.isdir(d) else "⚠️"
        print(f"  {ok} {name}: .../{Path(d).relative_to(_ROOT)}")

    print("\n[ Discrete Loaders ]")
    for name, fn in [
        ("Knapsack",  load_knapsack_cases),
        ("Graph",     load_graph_cases),
        ("TSP",       load_tsp_cases),
        ("SP-Unwt",   load_sp_unweighted_cases),
        ("SP-Wt",     load_sp_weighted_cases),
    ]:
        try:
            cases = fn()
            print(f"  ✅ {name}: {len(cases)} cases — first id={cases[0].id if cases else 'N/A'}")
        except Exception as e:
            print(f"  ❌ {name}: {e}")

    print("\n[ Continuous Loader ]")
    cont = load_cont_cases()
    print(f"  ✅ {len(cont)} combinations: {[(c.func_name, c.dim) for c in cont[:4]]}...")
    print(f"  CONT_ALGS = {CONT_ALGS}")
    print(f"  CONT_RUNS = {CONT_RUNS}")
    print("Done.")
