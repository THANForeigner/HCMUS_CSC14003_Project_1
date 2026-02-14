from __future__ import annotations
import os
import sys
import time
import math
import tracemalloc
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Any

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'nature_inspire', 'evolution_based', 'genetic_algorithm'))

try:
    from nature_inspire.biology_based.ACO_TSP import ACO_TSP
    from classical.informed.A_star_TSP import AStarTSP
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        from nature_inspire.biology_based.ACO_TSP import ACO_TSP
        from classical.informed.A_star_TSP import AStarTSP
    except ImportError as e:
        print(f"Algorithm Import Error: {e}")
try:
    from traveling_salesman_problem import GeneticAlgorithmTSP
except ImportError:
    try:
        from nature_inspire.evolution_based.genetic_algorithm.traveling_salesman_problem import GeneticAlgorithmTSP
    except ImportError as e:
         print(f"GA Import Error: {e}")
         pass
 

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not found. Plotting disabled.")

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.join(BASE_DIR, "tests_tsp")
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
    n = len(points)
    if n > 50: # A* is too slow for N > 50
        return None
        
    matrix = calculate_distance_matrix(points)
    solver = AStarTSP(matrix)
    try:
        cost, path = solver.solve()
        return cost, path
    except Exception as e:
        print(f"A* Error: {e}")
        return None

def run_aco_wrapper(points: List[Tuple[float, float]]) -> Optional[Tuple[float, List[int]]]:
    matrix = calculate_distance_matrix(points)
    # n_ants=20, max_iter=50 for reasonable speed
    solver = ACO_TSP(matrix, n_ants=20, max_iter=50)
    try:
        cost, path = solver.solve()
        return cost, path
    except Exception as e:
        print(f"ACO Error: {e}")
        return None

def run_ga_wrapper(points: List[Tuple[float, float]]) -> Optional[Tuple[float, List[int]]]:
    n = len(points)
    if n > 100:
        return None
    matrix = calculate_distance_matrix(points)
    # Correct instantiation based on existing code logic
    solver = GeneticAlgorithmTSP(matrix, pop_size=50, generations=100, mutation_rate=0.1)
    try:
        # GeneticAlgorithm.run() returns (best_individual, history)
        # best_individual is (path, fitness)
        res, history = solver.run()
        path, fitness = res
        cost = calculate_path_cost(path, matrix)
        return cost, path
    except Exception as e:
        print(f"GA Error: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- BENCHMARK ---
def bench_series(name: str, runner: Callable[[List[Tuple[float, float]]], Any], tests_dir: str, num_cases: int) -> AlgoSeries:
    recs: List[Record] = []
    
    # Get all .txt files in tests_dir and sort them numerically
    try:
        files = [f for f in os.listdir(tests_dir) if f.endswith('.txt')]
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
        test_ids = [n for n in sorted_nums if n <= num_cases]
    except FileNotFoundError:
        print(f"Tests directory not found: {tests_dir}")
        return AlgoSeries(name, [])

    for i in test_ids:
        in_path = os.path.join(tests_dir, f"{i}.txt")
        ans_path = os.path.join(tests_dir, f"{i}.ans")
        
        print(f"  Running Test {i} ({name})...", end="", flush=True)
        
        points = load_case(in_path)
        if not points:
             print(" -> Empty/Invalid input")
             continue
             
        expected_cost = load_ans(ans_path)
        
        tracemalloc.start()
        start_time = time.perf_counter()
        
        result = runner(points)
        
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

# --- PLOTTING ---
def plot_results(series_list: List[AlgoSeries]):
    if not MATPLOTLIB_AVAILABLE or not SAVE_PLOTS:
        return
        
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # Filter out empty series
    series_list = [s for s in series_list if s.records]
    if not series_list: return

    # 1. Execution Time
    plt.figure(figsize=(10, 6))
    for s in series_list:
        ids = [r.test_id for r in s.records if r.time_sec is not None]
        times = [r.time_sec for r in s.records if r.time_sec is not None]
        plt.plot(ids, times, marker="o", label=s.name)
    
    plt.title("TSP Execution Time (s)")
    plt.xlabel("Test Case ID")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, "tsp_time.png"), dpi=DPI)
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

    plt.title("TSP Path Cost")
    plt.xlabel("Test Case ID")
    plt.ylabel("Cost")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, "tsp_cost.png"), dpi=DPI)
    plt.close()

def main():
    print(f"Starting TSP Benchmark...")
    print(f"Test Directory: {TESTS_DIR}")
    
    # Run A*
    print("\n--- Benchmarking A* (Small Cases Only) ---")
    astar_series = bench_series("A*", run_astar_wrapper, TESTS_DIR, NUM_CASES)
    
    # Run ACO
    print("\n--- Benchmarking ACO ---")
    aco_series = bench_series("ACO", run_aco_wrapper, TESTS_DIR, NUM_CASES)
    
    # Run GA
    print("\n--- Benchmarking GA ---")
    ga_series = bench_series("GA", run_ga_wrapper, TESTS_DIR, NUM_CASES)
    
    # Plot
    plot_results([astar_series, aco_series, ga_series])
    print("\nBenchmark Completed.")

if __name__ == "__main__":
    main()
