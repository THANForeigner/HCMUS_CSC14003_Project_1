from __future__ import annotations
import os
import sys
import time
import tracemalloc
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Any

# Imports from local modules
try:
    from classical.uninformed.dfsColoring import DFSColoring
    from nature_inspire.biology_based.ACOColoring import ACOColoring
except ImportError:
    # Handle running from root directory
    sys.path.append(os.path.dirname(__file__))
    from classical.uninformed.dfsColoring import DFSColoring
    from nature_inspire.biology_based.ACOColoring import ACOColoring

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not found. Plotting disabled.")

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.join(BASE_DIR, "tests_graph_coloring")
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
    # Check max degree for upper bound if needed, but we start low.
    for k in range(1, n + 1):
        solver = DFSColoring(adj, n, k)
        res = solver.solve()
        if res:
            # res is colors array (size n+1)
            return k, res
    return None

def run_aco_wrapper(n: int, edges: List[Tuple[int, int]]) -> Optional[Tuple[int, List[int]]]:
    adj_dict = build_adj_dict(n, edges)
    # Params tuned for speed vs quality
    solver = ACOColoring(adj_dict, n, max_iter=30, n_ants=10)
    try:
        num_colors, colors_list = solver.solve()
        return num_colors, colors_list
    except Exception as e:
        print(f"ACO Error: {e}")
        return None

# --- BENCHMARK ---
def bench_series(name: str, runner: Callable[[int, List[Tuple[int, int]]], Any], tests_dir: str, num_cases: int) -> AlgoSeries:
    recs: List[Record] = []
    for i in range(1, num_cases + 1):
        in_path = os.path.join(tests_dir, f"{i}.txt")
        ans_path = os.path.join(tests_dir, f"{i}.ans")
        
        if not os.path.exists(in_path):
            continue
        print(f"  Running Test {i} ({name})...", end="", flush=True)
        
        n, edges = load_case(in_path)
        expected_k = load_ans(ans_path)
        tracemalloc.start()
        start_time = time.perf_counter()
        result = runner(n, edges)
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

# --- PLOTTING ---
def plot_results(series_list: List[AlgoSeries]):
    if not MATPLOTLIB_AVAILABLE or not SAVE_PLOTS:
        return
        
    os.makedirs(PLOT_DIR, exist_ok=True)
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
    plt.savefig(os.path.join(PLOT_DIR, "gc_time_memory.png"))
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
    plt.savefig(os.path.join(PLOT_DIR, "gc_quality_error.png"))
    plt.close()

def main():
    print(f"Starting Graph Coloring Benchmark (Cases 1-{NUM_CASES})...")
    
    # Run DFS
    print("\n--- Benchmarking DFS ---")
    dfs_series = bench_series("DFS", run_dfs_wrapper, TESTS_DIR, NUM_CASES)
    
    # Run ACO
    print("\n--- Benchmarking ACO ---")
    aco_series = bench_series("ACO", run_aco_wrapper, TESTS_DIR, NUM_CASES)
    
    # Plot
    plot_results([dfs_series, aco_series])
    print("\nBenchmark Completed.")

if __name__ == "__main__":
    main()
