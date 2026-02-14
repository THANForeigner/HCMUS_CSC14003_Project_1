from __future__ import annotations
import os
import time
import tracemalloc
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Any

# --- IMPORTS ---
from classical.uninformed.BFS import BFS
from classical.uninformed.DFS import DFS
from classical.uninformed.UCS import UCS
from classical.informed.greedy_best_first import GBFS
from nature_inspire.biology_based.ACO import ACO

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not found. Plotting disabled.")

# --- CONFIG ---
UNWEIGHTED_DIR = "tests_shortest_unweighted"
WEIGHTED_DIR = "tests_shortest_weighted"
NUM_CASES = 35
SAVE_PLOTS = True and MATPLOTLIB_AVAILABLE
PLOTS_DIR = "plots"
DPI = 220


# --- DATA STRUCTURES ---
@dataclass
class Record:
    test_id: int
    # Cho phép None để biểu thị test case bị bỏ qua (skipped)
    time_sec: Optional[float]
    peak_mem_mb: Optional[float]
    pct_error: Optional[float]


@dataclass
class AlgoSeries:
    name: str
    algo_type: str
    records: List[Record]


# --- DATA LOADING ---
def load_unweighted_case(path: str) -> Tuple[int, int, int, List[Tuple[int, int]]]:
    with open(path, "r", encoding="utf-8") as f:
        n, m = map(int, f.readline().split())
        s, t = map(int, f.readline().split())
        edges = [tuple(map(int, f.readline().split())) for _ in range(m)]
    return n, s, t, edges


def load_weighted_case(path: str) -> Tuple[int, int, int, List[Tuple[int, int, int]]]:
    with open(path, "r", encoding="utf-8") as f:
        n, m = map(int, f.readline().split())
        s, t = map(int, f.readline().split())
        edges = [tuple(map(int, f.readline().split())) for _ in range(m)]
    return n, s, t, edges


def load_ans(path: str) -> Optional[Tuple[int, List[int]]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            first = f.readline().strip()
            if first == "-1":
                return None
            x = int(first)
            line2 = f.readline().strip()
            nodes = list(map(int, line2.split())) if line2 else []
            return x, nodes
    except FileNotFoundError:
        return None


# --- GRAPH BUILDING ---
def build_adj_unweighted(n: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
    adj = [[] for _ in range(n + 1)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def build_adj_weighted(n: int, edges: List[Tuple[int, int, int]]) -> List[List[Tuple[int, int]]]:
    adj = [[] for _ in range(n + 1)]
    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))
    return adj


# --- VALIDATION HELPER ---
def is_valid_path_unweighted(adj: List[List[int]], path: List[int], s: int, t: int) -> bool:
    if not path or path[0] != s or path[-1] != t:
        return False
    for a, b in zip(path, path[1:]):
        if b not in adj[a]:
            return False
    return True


def path_cost_weighted(adjw: List[List[Tuple[int, int]]], path: List[int]) -> Optional[int]:
    if not path:
        return None
    total = 0
    for a, b in zip(path, path[1:]):
        found = False
        for v, w in adjw[a]:
            if v == b:
                total += w
                found = True
                break
        if not found:
            return None
    return total


def is_valid_path_weighted(adjw: List[List[Tuple[int, int]]], path: List[int], s: int, t: int) -> bool:
    if not path or path[0] != s or path[-1] != t:
        return False
    return path_cost_weighted(adjw, path) is not None


def percent_error(expected: Optional[int], got: Optional[int], expected_is_impossible: bool,
                  got_is_impossible: bool) -> float:
    if expected_is_impossible:
        return 0.0 if got_is_impossible else 100.0
    if got_is_impossible:
        return 100.0
    if expected is None or got is None:
        return 100.0
    if expected == 0:
        return 0.0 if got == 0 else 100.0
    return abs(got - expected) / abs(expected) * 100.0


# --- ALGO WRAPPERS ---
UnweightedAlgo = Callable[[List[List[int]], int, int], Optional[List[int]]]
WeightedAlgo = Callable[[List[List[Tuple[int, int]]], int, int], Any]


def run_aco_wrapper(adjw: List[List[Tuple[int, int]]], s: int, t: int) -> Any:
    graph_dict = {}
    n = len(adjw) - 1

    # Init dict
    for i in range(1, n + 1):
        graph_dict[str(i)] = {}

    for u in range(1, n + 1):
        for v, w in adjw[u]:
            u_str, v_str = str(u), str(v)
            if v_str in graph_dict[u_str]:
                graph_dict[u_str][v_str] = min(graph_dict[u_str][v_str], w)
            else:
                graph_dict[u_str][v_str] = w

    # Khởi tạo ACO - Có thể raise ValueError nếu N > Limit
    aco_solver = ACO(
        graph_dict=graph_dict,
        n_nodes=n,
        alpha=1.0,
        beta=2.0,
        rho=0.1,
        q=100.0
    )

    # Chạy thuật toán
    path, cost = aco_solver.run(str(s), str(t), n_ants=20, max_iter=20)

    if path is None:
        return None

    path_int = [int(node) for node in path]
    return (cost, path_int)


def get_unweighted_algorithms():
    return {"BFS": BFS, "DFS": DFS}


def get_weighted_algorithms():
    return {"UCS": UCS, "GBFS": GBFS, "ACO": run_aco_wrapper}


# --- BENCHMARK FUNCTIONS ---
def bench_unweighted_series(name: str, algo: UnweightedAlgo, tests_dir: str, num_cases: int) -> AlgoSeries:
    recs: List[Record] = []
    for i in range(1, num_cases + 1):
        in_path = os.path.join(tests_dir, f"{i}.txt")
        ans_path = os.path.join(tests_dir, f"{i}.ans")

        if not os.path.exists(in_path): continue

        tracemalloc.start()
        t0 = time.perf_counter()

        n, s, t, edges = load_unweighted_case(in_path)
        expected = load_ans(ans_path)
        exp_len = None if expected is None else expected[0]
        adj = build_adj_unweighted(n, edges)

        got_path = algo(adj, s, t)

        dt = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        exp_impossible = (expected is None)
        got_impossible = (got_path is None)

        got_len: Optional[int] = None
        if got_path is not None and is_valid_path_unweighted(adj, got_path, s, t):
            got_len = len(got_path)
        else:
            if got_path is not None: got_impossible = True

        err = percent_error(exp_len, got_len, exp_impossible, got_impossible)
        recs.append(Record(i, dt, peak / (1024 * 1024), err))

    return AlgoSeries(name=name, algo_type="unweighted", records=recs)


def bench_weighted_series(name: str, algo: WeightedAlgo, tests_dir: str, num_cases: int) -> AlgoSeries:
    recs: List[Record] = []

    for i in range(1, num_cases + 1):
        in_path = os.path.join(tests_dir, f"{i}.txt")
        ans_path = os.path.join(tests_dir, f"{i}.ans")

        if not os.path.exists(in_path):
            print(f"File not found: {in_path}")
            continue

        # Default values for skipped/failed tests
        dt = None
        peak_mb = None
        err = None

        skip_this_test = False

        try:
            tracemalloc.start()
            t0 = time.perf_counter()

            # IO + Parse
            n, s, t, edges = load_weighted_case(in_path)
            expected = load_ans(ans_path)
            exp_cost = None if expected is None else expected[0]
            adjw = build_adj_weighted(n, edges)

            # Build & Run Algo
            got = algo(adjw, s, t)  # Có thể raise ValueError tại đây

            # Stop Timer
            dt = time.perf_counter() - t0
            _, peak = tracemalloc.get_traced_memory()
            peak_mb = peak / (1024 * 1024)
            tracemalloc.stop()

            # --- Validation Logic (Chỉ chạy khi không lỗi) ---
            exp_impossible = (expected is None)
            got_impossible = (got is None)
            got_cost: Optional[int] = None
            got_path: Optional[List[int]] = None

            if got is None:
                pass
            elif isinstance(got, tuple) and len(got) == 2:
                got_cost = int(got[0])
                got_path = got[1]
            elif isinstance(got, list):
                got_path = got
                pc = path_cost_weighted(adjw, got_path)
                got_cost = int(pc) if pc is not None else None

            if got_path is not None and got_cost is not None:
                if not is_valid_path_weighted(adjw, got_path, s, t):
                    got_impossible = True
                    got_cost = None
            else:
                if got is not None: got_impossible = True

            err = percent_error(exp_cost, got_cost, exp_impossible, got_impossible)

        except ValueError as e:
            # Bắt lỗi "Graph quá lớn" từ ACO
            tracemalloc.stop()  # Dừng trace nếu đang chạy
            print(f"  [SKIP] Test {i} ({name}): {e}")
            skip_this_test = True

        except Exception as e:
            tracemalloc.stop()
            print(f"  [ERROR] Test {i} ({name}): {e}")
            skip_this_test = True

        # Nếu skip_this_test = True, thì dt, peak_mb, err vẫn là None
        recs.append(Record(i, dt, peak_mb, err))

    return AlgoSeries(name=name, algo_type="weighted", records=recs)


# --- PLOTTING ---
def ensure_plots_dir():
    if SAVE_PLOTS:
        os.makedirs(PLOTS_DIR, exist_ok=True)


def savefig(filename: str):
    if SAVE_PLOTS:
        plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=DPI, bbox_inches="tight")


def plot_time_memory_per_test(series_list: List[AlgoSeries], num_cases: int):
    if not MATPLOTLIB_AVAILABLE: return
    xs = list(range(1, num_cases + 1))
    fig, (ax_time, ax_mem) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for s in series_list:
        # Matplotlib tự động bỏ qua các giá trị None (làm đứt nét vẽ)
        times = [r.time_sec for r in s.records]
        mems = [r.peak_mem_mb for r in s.records]

        ax_time.plot(xs, times, marker="o", linewidth=1, label=f"{s.algo_type}:{s.name}")
        ax_mem.plot(xs, mems, marker="o", linewidth=1, label=f"{s.algo_type}:{s.name}")

    ax_time.set_title("Test vs Time (s)")
    ax_time.legend()
    ax_mem.set_title("Test vs Peak Memory (MB)")
    ax_mem.legend()
    ax_mem.set_xticks(xs)
    fig.tight_layout()
    savefig("pertest_time_memory.png")


def plot_error_per_test(series_list: List[AlgoSeries], num_cases: int):
    if not MATPLOTLIB_AVAILABLE: return
    xs = list(range(1, num_cases + 1))
    plt.figure(figsize=(12, 5))
    for s in series_list:
        # Bỏ qua DFS nếu muốn vì nó không tối ưu
        if s.name == "DFS": continue

        errs = [r.pct_error for r in s.records]
        plt.plot(xs, errs, marker="o", linewidth=1, label=f"{s.algo_type}:{s.name}")

    plt.title("Test vs % Error")
    plt.xlabel("Test ID")
    plt.ylabel("% Error")
    plt.xticks(xs)
    plt.legend()
    plt.tight_layout()
    savefig("pertest_percent_error.png")


# --- MAIN ---
def main():
    ensure_plots_dir()
    all_series: List[AlgoSeries] = []

    # 1. Unweighted
    for name, algo in get_unweighted_algorithms().items():
        print(f"Benchmark UNWEIGHTED {name} ...")
        all_series.append(bench_unweighted_series(name, algo, UNWEIGHTED_DIR, NUM_CASES))

    # 2. Weighted
    for name, algo in get_weighted_algorithms().items():
        print(f"Benchmark WEIGHTED {name} ...")
        all_series.append(bench_weighted_series(name, algo, WEIGHTED_DIR, NUM_CASES))

    # 3. Plots
    plot_time_memory_per_test(all_series, NUM_CASES)
    plot_error_per_test(all_series, NUM_CASES)

    if SAVE_PLOTS:
        print(f"Saved images to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()