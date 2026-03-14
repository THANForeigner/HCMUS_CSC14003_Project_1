from __future__ import annotations
import os
import sys
import signal

# Ensure project root and necessary module paths are in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'nature_inspire', 'evolution_based', 'genetic_algorithm'))

import time
import math
import tracemalloc
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Any

from classical.informed.A_star_TSP import AStarTSP
from classical.local.hill_climbing_tsp import HillClimbingTSP
from nature_inspire.biology_based.ant_colony_optimization.ant_colony_optimization_tsp import ACO_TSP
from nature_inspire.evolution_based.genetic_algorithm.genetic_algorithm_tsp import GA_TSP
from nature_inspire.physic_based.simulated_annealing.simulated_annealing_tsp import SA_TSP
 

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
    from problems.input_validator import load_tsp_cases, TSP_DIR
except ImportError as _e:
    print(f"Warning: input_validator not found ({_e}).")
    load_tsp_cases = None; TSP_DIR = None

# --- TIMEOUT HANDLER ---
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = TSP_DIR or os.path.join(BASE_DIR, "tests_tsp")
PLOT_DIR = os.path.join(BASE_DIR, "plots")
NUM_CASES = 16 
SAVE_PLOTS = True and MATPLOTLIB_AVAILABLE
DPI = 220

# --- DATA STRUCTURES ---
@dataclass
class Record:
    test_id: int
    time_sec: Optional[float]
    peak_mem_mb: Optional[float]
    cost: Optional[float]
    expected_cost: Optional[float]
    pct_error: Optional[float]

@dataclass
class AlgoSeries:
    name: str
    records: List[Record]

# --- DATA LOADING ---
def load_case(path: str) -> List[Tuple[float, float]]:
    points = []
    with open(path, "r", encoding="utf-8") as f:
        line = f.readline()
        if not line: return []
        try:
            n = int(line.strip())
        except ValueError:
            return []
            
        for _ in range(n):
            try:
                parts = list(map(float, f.readline().split()))
                if len(parts) >= 2:
                    points.append((parts[0], parts[1]))
            except ValueError:
                continue
    return points

def load_ans(path: str) -> Optional[float]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            line = f.readline().strip()
            if not line: return None
            val = float(line)
            if val == -1: return None
            return val
    except (FileNotFoundError, ValueError):
        return None

def calculate_distance_matrix(points: List[Tuple[float, float]]) -> List[List[float]]:
    n = len(points)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist = math.sqrt((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)
            matrix[i][j] = dist
    return matrix

def calculate_path_cost(path: List[int], matrix: List[List[float]]) -> float:
    cost = 0.0
    n = len(path)
    if n == 0: return 0.0
    for i in range(n):
        u = path[i]
        v = path[(i + 1) % n]
        cost += matrix[u][v]
    return cost

# --- ALGO WRAPPERS ---
# Return type: (cost, path)
def run_astar_wrapper(points: List[Tuple[float, float]]) -> Optional[Tuple[float, List[int]]]:
    pts, matrix = points, calculate_distance_matrix(points)
    n = len(pts)
    if n > 20:
        print(f" [A* Skipped: N={n} too large for Exact Solver]")
        return None
    solver = AStarTSP(matrix)
    try:
        cost, path = solver.run()
        return cost, path
    except Exception as e:
        print(f"A* Error: {e}")
        return None

def run_ga_wrapper(points: List[Tuple[float, float]]) -> Optional[Tuple[float, List[int]]]:
    pts, matrix = points, calculate_distance_matrix(points)
    n = len(pts)
    if n > 200:
        print(f" [GA Skipped: N={n} exceeds benchmark limit]")
        return None
    config = algo_config.get("GA", {})
    pop_size = config.get("population_size", 100 if n > 50 else 50)
    generations = config.get("max_iter", 200 if n > 50 else 100)
    solver = GA_TSP(matrix, pop_size=pop_size, generations=generations)
    try:
        res, history = solver.run()
        path, fitness = res
        cost = sum(matrix[path[i]][path[(i+1)%n]] for i in range(n))
        return cost, path
    except Exception as e:
        print(f"GA Error: {e}")
        return None


def run_aco_wrapper(points: List[Tuple[float, float]]) -> Optional[Tuple[float, List[int]]]:
    pts, matrix = points, calculate_distance_matrix(points)
    n = len(pts)
    if n > 250:
        print(f" [ACO Skipped: N={n} too large for quick benchmark]")
        return None
    config = algo_config.get("ACO", {})
    n_ants = config.get("n_ants", 20 if n < 100 else 30)
    max_iter = config.get("max_iter", 50 if n < 100 else 100)
    solver = ACO_TSP(matrix, n_ants=n_ants, max_iter=max_iter)
    try:
        cost, path = solver.run()
        return cost, path
    except Exception as e:
        print(f"ACO Error: {e}")
        return None


def run_hill_climbing_wrapper(points: List[Tuple[float, float]]) -> Optional[Tuple[float, List[int]]]:
    pts, matrix = points, calculate_distance_matrix(points)
    n = len(pts)
    if n > 300:
        return None
    config = algo_config.get("HC", {})
    restarts = 20 if n < 50 else 5
    solver = HillClimbingTSP(matrix, max_iterations=500, restarts=restarts)
    try:
        cost, path = solver.solve()
        return cost, path
    except Exception:
        return None

def run_sa_wrapper(points: List[Tuple[float, float]]) -> Optional[Tuple[float, List[int]]]:
    pts, matrix = points, calculate_distance_matrix(points)
    n = len(pts)
    if n > 300:
        return None
    config = algo_config.get("SA", {})
    initial_temp = config.get("initial_temp", 100.0)
    alpha = config.get("alpha", 0.95)
    stopping_T = config.get("final_temp", 0.001)

    solver = SA_TSP(max_vertices=n, coords=points, T=initial_temp, stopping_T=stopping_T, alpha=alpha)
    try:
        # Scale runs with problem size
        n_times = 5 if n < 100 else 2
        solver.run(times=n_times)
        return solver.best_solution_result, solver.best_solution_nodes
    except Exception as e:
        print(f"SA Error: {e}")
        return None

# --- BENCHMARK CLASS ---
class TSPBenchmark:
    def __init__(self, tests_dir: str = TESTS_DIR, plot_dir: str = PLOT_DIR, num_cases: int = NUM_CASES, save_plots: bool = SAVE_PLOTS, dpi: int = DPI):
        self.tests_dir = tests_dir
        self.plot_dir = plot_dir
        self.num_cases = num_cases
        self.save_plots = save_plots
        if not MATPLOTLIB_AVAILABLE:
            self.save_plots = False
        self.dpi = dpi

    def bench_series(self, name: str, runner: Callable[[List[Tuple[float, float]]], Any]) -> AlgoSeries:
        recs: List[Record] = []
        
        # Get all .txt files in tests_dir and sort them numerically
        try:
            files = [f for f in os.listdir(self.tests_dir) if f.endswith('.txt')]
            # Extract numbers to sort
            files_map = {}
            for f in files:
                try:
                    num = int(os.path.splitext(f)[0])
                    files_map[num] = f
                except ValueError:
                    pass
            
            sorted_nums = sorted(files_map.keys())
            # Use provided NUM_CASES to slice if needed, or just all
            # We want exactly the cases 1..NUM_CASES or whatever is available
            test_ids = [n for n in sorted_nums if n <= self.num_cases]
        except FileNotFoundError:
            print(f"Tests directory not found: {self.tests_dir}")
            return AlgoSeries(name, [])

        cases = load_tsp_cases(tests_dir=self.tests_dir, num=self.num_cases) if load_tsp_cases else []
        for case in cases:
            i, points, expected_cost = case
            print(f"  Running Test {i} ({name})...", end="", flush=True)
            tracemalloc.start()
            start_time = time.perf_counter()
            result = None
            got_cost = None
            pct_error = 0.0
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)
            
            try:
                result = runner(points)
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
            
            got_cost = None
            pct_error = 0.0
            
            if result:
                got_cost, path = result
                if got_cost == float('inf') or got_cost is None:
                     print(f" -> NO SOLUTION!", end="")
                     got_cost = None
                else:
                     if expected_cost:
                         pct_error = (got_cost - expected_cost) / expected_cost * 100.0
                     else:
                         pct_error = 0.0
                     print(f" Done. Time: {dt:.4f}s, Cost: {got_cost:.2f} (Exp: {expected_cost})")
            else:
                print(f" -> SKIPPED/FAILED!", end="")
                print(f" Done. Time: {dt:.4f}s")
            
            recs.append(Record(i, dt, peak_mb, got_cost, expected_cost, pct_error))
            
        return AlgoSeries(name, recs)

    def plot_results(self, series_list: List[AlgoSeries]):
        if not MATPLOTLIB_AVAILABLE or not self.save_plots:
            return
            
        os.makedirs(self.plot_dir, exist_ok=True)
        
        # Filter out empty series
        series_list = [s for s in series_list if s.records]
        if not series_list: return

        # 1. Execution Time
        plt.figure(figsize=(10, 6))
        for s in series_list:
            ids = [r.test_id for r in s.records if r.time_sec is not None]
            times = [r.time_sec for r in s.records if r.time_sec is not None]
            plt.plot(ids, times, marker="o", label=s.name)
        
        plt.title("Traveling Salesman Problem (TSP) - Execution Time (s)")
        plt.xlabel("Test Case ID")
        plt.ylabel("Time (s)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plot_dir, "tsp_time.png"), dpi=self.dpi)
        plt.close()

        # 2. Cost Comparison
        plt.figure(figsize=(10, 6))
        for s in series_list:
            ids = [r.test_id for r in s.records if r.cost is not None]
            costs = [r.cost for r in s.records if r.cost is not None]
            plt.plot(ids, costs, marker="x", label=s.name)
        
        # Also plot expected if available (from first series that has it)
        if series_list:
            ref = series_list[0]
            ids = [r.test_id for r in ref.records if r.expected_cost is not None]
            exp = [r.expected_cost for r in ref.records if r.expected_cost is not None]
            if ids:
                plt.plot(ids, exp, "k--", label="Optimal (DP)")

        plt.title("Traveling Salesman Problem (TSP) - Path Cost")
        plt.xlabel("Test Case ID")
        plt.ylabel("Cost")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plot_dir, "tsp_cost.png"), dpi=self.dpi)
        plt.close()

    def run(self):
        print(f"Starting TSP Benchmark...")
        print(f"Test Directory: {self.tests_dir}")
        
        # --- WARM-UP: run each algorithm on a trivial input to preload libraries/JIT ---
        print("Warming up algorithms (eliminating cold start)...", end="", flush=True)
        import contextlib, io
        _dummy_pts = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]  # 3-node triangle
        with contextlib.redirect_stdout(io.StringIO()):
            try: run_astar_wrapper(_dummy_pts)
            except Exception: pass
            try: run_aco_wrapper(_dummy_pts)
            except Exception: pass
            try: run_ga_wrapper(_dummy_pts)
            except Exception: pass
            try: run_hill_climbing_wrapper(_dummy_pts)
            except Exception: pass
            try: run_sa_wrapper(_dummy_pts)
            except Exception: pass
        print(" Done.")
        
        # Run A*
        print("\n--- Benchmarking A* (Small Cases Only) ---")
        astar_series = self.bench_series("A*", run_astar_wrapper)
        
        # Run ACO
        print("\n--- Benchmarking ACO ---")
        aco_series = self.bench_series("ACO", run_aco_wrapper)
        
        # Run GA
        print("\n--- Benchmarking GA ---")
        ga_series = self.bench_series("GA", run_ga_wrapper)
        
        # Run Hill Climbing
        print("\n--- Benchmarking Hill Climbing ---")
        hc_series = self.bench_series("Hill Climbing", run_hill_climbing_wrapper)
        
        # Run Simulated Annealing
        print("\n--- Benchmarking Simulated Annealing ---")
        sa_series = self.bench_series("Simulated Annealing", run_sa_wrapper)
        
        # Plot
        self.plot_results([astar_series, aco_series, ga_series, hc_series, sa_series])
        print("\nBenchmark Completed.")

def main():
    benchmark = TSPBenchmark()
    benchmark.run()

if __name__ == "__main__":
    main()
