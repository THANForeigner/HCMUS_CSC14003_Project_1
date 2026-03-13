from __future__ import annotations
import os
import sys
import signal
import time
import tracemalloc
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Any

# --- IMPORTS ---
from classical.uninformed.breath_first_search_shortest_path import BFS
from classical.uninformed.depth_first_search_shortest_path import DFS
from classical.uninformed.uniform_cost_search import UCS
from classical.informed.greedy_best_first_search_shortest_path import GBFS
from nature_inspire.biology_based.ant_colony_optimization.ant_colony_optimization import ACO

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not found. Plotting disabled.")

try:
    from nature_inspire.problem import algo_config
except ImportError:
    try:
        from problems.problem import algo_config
    except ImportError:
        algo_config = {}

try:
    from problems.input_validator import (
        load_sp_unweighted_cases, load_sp_weighted_cases,
        SP_UNWEIGHTED_DIR, SP_WEIGHTED_DIR,
    )
except ImportError as _e:
    print(f"Warning: input_validator not found ({_e}).")
    load_sp_unweighted_cases = load_sp_weighted_cases = None
    SP_UNWEIGHTED_DIR = SP_WEIGHTED_DIR = None

# --- TIMEOUT HANDLER ---
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UNWEIGHTED_DIR  = SP_UNWEIGHTED_DIR or os.path.join(BASE_DIR, "tests_shortest_unweighted")
WEIGHTED_DIR    = SP_WEIGHTED_DIR   or os.path.join(BASE_DIR, "tests_shortest_weighted")
NUM_CASES = 35
SAVE_PLOTS = True and MATPLOTLIB_AVAILABLE
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
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

    config = algo_config.get("ACO", {})
    n_ants = config.get("n_ants", 20)
    max_iter = config.get("max_iter", 20)

    # Chạy thuật toán
    path, cost = aco_solver.run(str(s), str(t), n_ants=n_ants, max_iter=max_iter)

    if path is None:
        return None

    path_int = [int(node) for node in path]
    return (cost, path_int)


def get_unweighted_algorithms():
    return {"BFS": BFS, "DFS": DFS}


def get_weighted_algorithms():
    return {"UCS": UCS, "GBFS": GBFS, "ACO": run_aco_wrapper}


# --- BENCHMARK CLASS ---
class ShortestPathBenchmark:
    def __init__(self, unweighted_dir: str = UNWEIGHTED_DIR, weighted_dir: str = WEIGHTED_DIR, plots_dir: str = PLOTS_DIR, num_cases: int = NUM_CASES, save_plots: bool = SAVE_PLOTS, dpi: int = DPI):
        self.unweighted_dir = unweighted_dir
        self.weighted_dir = weighted_dir
        self.plots_dir = plots_dir
        self.num_cases = num_cases
        self.save_plots = save_plots# Only true if also MATPLOTLIB_AVAILABLE globally
        if not MATPLOTLIB_AVAILABLE:
            self.save_plots = False
        self.dpi = dpi

    def bench_unweighted_series(self, name: str, algo: UnweightedAlgo) -> AlgoSeries:
        recs: List[Record] = []
        cases = load_sp_unweighted_cases(tests_dir=self.unweighted_dir, num=self.num_cases) if load_sp_unweighted_cases else []
        for case in cases:
            i, n, s, t, edges, expected = case
            exp_len = None if expected is None else expected[0]
            adj = build_adj_unweighted(n, edges)
            got_path = None
            tracemalloc.start()
            t0 = time.perf_counter()
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)
            
            try:
                got_path = algo(adj, s, t)
                signal.alarm(0)  # Disable alarm
            except TimeoutException:
                print(f" -> TIMEOUT (1 min)!", end="")
                got_path = None
            except Exception as e:
                print(f" -> ERROR: {e}", end="")
                got_path = None
            finally:
                signal.alarm(0)

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


    def bench_weighted_series(self, name: str, algo: WeightedAlgo) -> AlgoSeries:
        recs: List[Record] = []
        cases = load_sp_weighted_cases(tests_dir=self.weighted_dir, num=self.num_cases) if load_sp_weighted_cases else []
        for case in cases:
            i, n, s, t, edges, expected = case
            exp_cost = None if expected is None else expected[0]
            dt = peak_mb = err = None
            skip_this_test = False
            try:
                tracemalloc.start()
                t0 = time.perf_counter()
                adjw = build_adj_weighted(n, edges)

                # Build & Run Algo
                got = None
                
                # Setup timeout
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(60)  # 1 minute timeout
                
                try:
                    got = algo(adjw, s, t)
                    signal.alarm(0)  # Disable alarm
                except TimeoutException:
                    print(f" -> TIMEOUT (1 min)!", end="")
                    got = None
                except Exception as e:
                    # Reraise ValueError for the existing skip logic
                    if isinstance(e, ValueError): raise e
                    print(f" -> ERROR: {e}", end="")
                    got = None
                finally:
                    signal.alarm(0)

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
    def ensure_plots_dir(self):
        if self.save_plots:
            os.makedirs(self.plots_dir, exist_ok=True)

    def savefig(self, filename: str):
        if self.save_plots:
            plt.savefig(os.path.join(self.plots_dir, filename), dpi=self.dpi, bbox_inches="tight")

    def plot_time_memory_per_test(self, series_list: List[AlgoSeries]):
        if not MATPLOTLIB_AVAILABLE: return
        xs = list(range(1, self.num_cases + 1))
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
        self.savefig("pertest_time_memory.png")
        plt.close()

    def plot_error_per_test(self, series_list: List[AlgoSeries]):
        if not MATPLOTLIB_AVAILABLE: return
        xs = list(range(1, self.num_cases + 1))
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
        self.savefig("pertest_percent_error.png")
        plt.close()

    def run(self):
        self.ensure_plots_dir()
        all_series: List[AlgoSeries] = []

        # --- WARM-UP: run each algorithm on a trivial graph to preload libraries/JIT ---
        print("Warming up algorithms (eliminating cold start)...", end="", flush=True)
        import contextlib, io
        _adj_uw = [[],[2],[1]]   # 2-node graph: 1 <-> 2
        _adjw = [[],[(2, 1)],[(1, 1)]]  # weighted version
        with contextlib.redirect_stdout(io.StringIO()):
            for algo in get_unweighted_algorithms().values():
                try: algo(_adj_uw, 1, 2)
                except Exception: pass
            for algo in get_weighted_algorithms().values():
                try: algo(_adjw, 1, 2)
                except Exception: pass
        print(" Done.")

        # 1. Unweighted
        for name, algo in get_unweighted_algorithms().items():
            print(f"Benchmark UNWEIGHTED {name} ...")
            all_series.append(self.bench_unweighted_series(name, algo))

        # 2. Weighted
        for name, algo in get_weighted_algorithms().items():
            print(f"Benchmark WEIGHTED {name} ...")
            all_series.append(self.bench_weighted_series(name, algo))

        # 3. Plots
        self.plot_time_memory_per_test(all_series)
        self.plot_error_per_test(all_series)

        if self.save_plots:
            print(f"Saved images to {self.plots_dir}/")


def main():
    benchmark = ShortestPathBenchmark()
    benchmark.run()

if __name__ == "__main__":
    main()