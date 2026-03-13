from __future__ import annotations
import os
import sys
import signal
import time
import tracemalloc
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Any
# Ensure project root and necessary module paths are in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'nature_inspire', 'evolution_based', 'genetic_algorithm'))

from classical.uninformed.depth_first_search_graph_coloring import DFS_GraphColoring
from nature_inspire.biology_based.ant_colony_optimization.ant_colony_optimization_graph_coloring import ACO_GraphColoring
from nature_inspire.evolution_based.genetic_algorithm.genetic_algorithm_graph_coloring import GA_GraphColoring
from nature_inspire.physic_based.simulated_annealing.simulated_annealing_graph_coloring import SA_GraphColoring
import numpy as np
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not found. Plotting disabled.")

try:
    from problems.problem import algo_config
except ImportError:
    algo_config = {}

try:
    from problems.input_validator import load_graph_cases, GRAPH_DIR
except ImportError as _e:
    print(f"Warning: input_validator not found ({_e}).")
    load_graph_cases = None; GRAPH_DIR = None

# --- TIMEOUT HANDLER ---
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = GRAPH_DIR or os.path.join(BASE_DIR, "tests_graph_coloring")
PLOT_DIR = os.path.join(BASE_DIR, "plots")
NUM_CASES = 12  # Adjust based on actual number of tests available or desired
SAVE_PLOTS = True and MATPLOTLIB_AVAILABLE
DPI = 220

# --- DATA STRUCTURES ---
@dataclass
class Record:
    test_id: int
    time_sec: Optional[float]
    peak_mem_mb: Optional[float]
    pct_error: Optional[float]
    num_colors: Optional[int]

@dataclass
class AlgoSeries:
    name: str
    records: List[Record]

# --- DATA LOADING ---
def load_case(path: str) -> Tuple[int, List[Tuple[int, int]]]:
    with open(path, "r", encoding="utf-8") as f:
        line1 = f.readline().split()
        if not line1: return 0, []
        n, m = map(int, line1)
        edges = []
        for _ in range(m):
            parts = list(map(int, f.readline().split()))
            if len(parts) >= 2:
                edges.append((parts[0], parts[1]))
    return n, edges

def load_ans(path: str) -> Optional[int]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            line = f.readline().strip()
            if not line: return None
            return int(line)
    except (FileNotFoundError, ValueError):
        return None

# --- GRAPH HELPER ---
def build_adj_list(n: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
    adj = [[] for _ in range(n + 1)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def build_adj_dict(n: int, edges: List[Tuple[int, int]]) -> Dict[str, List[str]]:
    # ACO expects dict with string keys/values according to ACOColoring.py (though it converts to int internally)
    d = {str(i): [] for i in range(1, n + 1)}
    for u, v in edges:
        d[str(u)].append(str(v))
        d[str(v)].append(str(u))
    return d

# --- VALIDATION ---
def is_valid_coloring(adj: List[List[int]], colors: List[int], n: int) -> bool:
    c_arr = colors
    if len(colors) == n:
        c_arr = [0] + colors
    
    for u in range(1, n + 1):
        if c_arr[u] == 0:
            return False
        for v in adj[u]:
            if c_arr[u] == c_arr[v]:
                return False
    return True

# --- ALGO WRAPPERS ---
# Return type: (num_colors, color_list)
# color_list should be valid for indices 1..N

def run_dfs_wrapper(n: int, edges: List[Tuple[int, int]]) -> Optional[Tuple[int, List[int]]]:
    adj = build_adj_list(n, edges)
    for k in range(1, n + 1):
        solver = DFS_GraphColoring(adj, n, k)
        res = solver.solve()
        if res:
            return k, res
    return None

def run_aco_wrapper(n: int, edges: List[Tuple[int, int]]) -> Optional[Tuple[int, List[int]]]:
    adj_dict = build_adj_dict(n, edges)
    config = algo_config.get("ACO", {})
    n_ants = config.get("n_ants", 10)
    max_iter = config.get("max_iter", 30)
    solver = ACO_GraphColoring(adj_dict, n, max_iter=max_iter, n_ants=n_ants)
    try:
        num_colors, colors_list = solver.run()
        return num_colors, colors_list
    except Exception as e:
        print(f"ACO Error: {e}")
        return None

def run_ga_wrapper(n: int, edges: List[Tuple[int, int]]) -> Optional[Tuple[int, List[int]]]:
    adj_matrix = np.zeros((n, n), dtype=int)
    for u, v in edges:
        adj_matrix[u-1][v-1] = 1
        adj_matrix[v-1][u-1] = 1

    config = algo_config.get("GA", {})
    pop_size = config.get("population_size", 50)
    max_iter = config.get("max_iter", 100)
    mutation_rate = config.get("mutation_rate", 0.1)
    crossover_rate = config.get("crossover_rate", 0.8)
    elite_size = config.get("elite_size", 2)
    for k in range(1, n + 1):
        solver = GA_GraphColoring(
            adj_matrix=adj_matrix, num_colors=k,
            pop_size=pop_size, generations=max_iter,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            elite_size=elite_size
        )
        try:
            (best_ind, best_fitness), history = solver.run()
            if best_fitness == 0:
                return k, [c + 1 for c in best_ind]
        except Exception as e:
            print(f"GA Error at K={k}: {e}")
            continue
    return None

def run_sa_wrapper(n: int, edges: List[Tuple[int, int]]) -> Optional[Tuple[int, List[int]]]:
    n_sa, zero_indexed_edges = n, [(u-1, v-1) for u, v in edges]

    config = algo_config.get("SA", {})
    T = config.get("initial_temp", 100.0)
    alpha = config.get("alpha", 0.95)
    stopping_T = config.get("final_temp", 0.001)
    for k in range(1, n_sa + 1):
        solver = SA_GraphColoring(
            max_colors=k, max_vertices=n_sa, edges=zero_indexed_edges,
            T=T, alpha=alpha, stopping_T=stopping_T
        )
        try:
            solver.run(times=10)
            if solver.best_energy == 0:
                return k, [c + 1 for c in solver.best_solution]
        except Exception as e:
            print(f"SA Error at K={k}: {e}")
            continue
    return None

# --- BENCHMARK CLASS ---
class GraphColoringBenchmark:
    def __init__(self, tests_dir: str = TESTS_DIR, plot_dir: str = PLOT_DIR, num_cases: int = NUM_CASES, save_plots: bool = SAVE_PLOTS, dpi: int = DPI):
        self.tests_dir = tests_dir
        self.plot_dir = plot_dir
        self.num_cases = num_cases
        self.save_plots = save_plots
        self.dpi = dpi

    def bench_series(self, name: str, runner: Callable[[int, List[Tuple[int, int]]], Any]) -> AlgoSeries:
        recs: List[Record] = []
        cases = load_graph_cases(tests_dir=self.tests_dir, num=self.num_cases) if load_graph_cases else []
        for case in cases:
            i, n, edges, expected_k = case
            print(f"  Running Test {i} ({name})...", end="", flush=True)
            tracemalloc.start()
            start_time = time.perf_counter()
            result = None
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)
            
            try:
                result = runner(n, edges)
                signal.alarm(0)  # Disable alarm
            except TimeoutException:
                print(f" -> TIMEOUT (1 min)!", end="")
                result = None
            except Exception as e:
                print(f" -> ERROR: {e}", end="")
                result = None
            finally:
                signal.alarm(0)

            dt = time.perf_counter() - start_time
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_mb = peak / (1024 * 1024)
            got_k = None
            pct_error = 0.0
            
            if result:
                got_k, colors = result
                adj = build_adj_list(n, edges)
                if len(colors) == n:
                    colors_check = [0] + colors
                else:
                    colors_check = colors
                if not is_valid_coloring(adj, colors_check, n):
                    print(f" -> INVALID COLORING!", end="")
                    got_k = None # Mark as failed
                else:
                    if expected_k:
                        pct_error = (got_k - expected_k) / expected_k * 100.0
                    else:
                        pct_error = 0.0
            else:
                print(f" -> NO SOLUTION!", end="")
                pct_error = 100.0
                
            print(f" Done. Time: {dt:.4f}s, K: {got_k} (Exp: {expected_k})")
            
            recs.append(Record(i, dt, peak_mb, pct_error, got_k))
            
        return AlgoSeries(name, recs)

    def plot_results(self, series_list: List[AlgoSeries]):
        if not MATPLOTLIB_AVAILABLE or not self.save_plots:
            return
            
        os.makedirs(self.plot_dir, exist_ok=True)
        # 1. Time & Memory
        fig, (ax_time, ax_mem) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        for s in series_list:
            ids = [r.test_id for r in s.records]
            times = [r.time_sec for r in s.records]
            mems = [r.peak_mem_mb for r in s.records]
            ax_time.plot(ids, times, marker="o", label=s.name)
            ax_mem.plot(ids, mems, marker="x", label=s.name)
            
        ax_time.set_title("Execution Time (s)")
        ax_time.set_ylabel("Seconds")
        ax_time.legend()
        
        ax_mem.set_title("Peak Memory (MB)")
        ax_mem.set_ylabel("MB")
        ax_mem.legend()
        ax_mem.set_xlabel("Test Case ID")
        
        fig.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "gc_time_memory.png"), dpi=self.dpi)
        plt.close()
        
        # 2. Results Quality (Num Colors)
        plt.figure(figsize=(10, 5))
        if series_list:
            for s in series_list:
                ids = [r.test_id for r in s.records]
                errs = [r.pct_error for r in s.records]
                plt.plot(ids, errs, marker="o", label=s.name)
                
        plt.title("Error % relative to Optimal/Expected K")
        plt.ylabel("% Error (Lower is Better)")
        plt.xlabel("Test Case ID")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "gc_quality_error.png"), dpi=self.dpi)
        plt.close()

    def run(self):
        print(f"Starting Graph Coloring Benchmark (Cases 1-{self.num_cases})...")
        
        # --- WARM-UP: run each algorithm on a trivial graph to preload libraries/JIT ---
        print("Warming up algorithms (eliminating cold start)...", end="", flush=True)
        _dummy_n = 2
        _dummy_edges = [(1, 2)]
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            try: run_dfs_wrapper(_dummy_n, _dummy_edges)
            except Exception: pass
            try: run_aco_wrapper(_dummy_n, _dummy_edges)
            except Exception: pass
            try: run_ga_wrapper(_dummy_n, _dummy_edges)
            except Exception: pass
            try: run_sa_wrapper(_dummy_n, _dummy_edges)
            except Exception: pass
        print(" Done.")
        
        # Run DFS
        print("\n--- Benchmarking DFS ---")
        dfs_series = self.bench_series("DFS", run_dfs_wrapper)
        
        # Run ACO
        print("\n--- Benchmarking ACO ---")
        aco_series = self.bench_series("ACO", run_aco_wrapper)
        
        # Run GA
        print("\n--- Benchmarking GA ---")
        ga_series = self.bench_series("GA", run_ga_wrapper)
        
        # Run SA
        print("\n--- Benchmarking SA ---")
        sa_series = self.bench_series("SA", run_sa_wrapper)
        
        # Plot
        self.plot_results([dfs_series, aco_series, ga_series, sa_series])
        print("\nBenchmark Completed.")

def main():
    benchmark = GraphColoringBenchmark()
    benchmark.run()

if __name__ == "__main__":
    main()
