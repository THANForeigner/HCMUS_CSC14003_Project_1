from __future__ import annotations
import os
import sys
import time
import tracemalloc
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Any

# Ensure we can import from parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from classical.informed.A_star_Knapsack import AStarKnapsack
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        from classical.informed.A_star_Knapsack import AStarKnapsack
    except ImportError as e:
        print(f"Import Error: {e}")

GeneticAlgorithmKnapsack = None
try:
    from nature_inspire.evolution_based.genetic_algorithm.knapsack import GeneticAlgorithmKnapsack
except ImportError as e:
    print(f"Warning: GeneticAlgorithmKnapsack not found. {e}")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not found. Plotting disabled.")

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.join(BASE_DIR, "tests_knapsack")
PLOT_DIR = os.path.join(BASE_DIR, "plots")
# Limit cases if needed, A* is exponential
NUM_CASES = 15
SAVE_PLOTS = True and MATPLOTLIB_AVAILABLE
DPI = 220

# --- DATA STRUCTURES ---
@dataclass
class Record:
    test_id: int
    time_sec: Optional[float]
    peak_mem_mb: Optional[float]
    value: Optional[float]
    expected_value: Optional[float]
    pct_error: Optional[float]

@dataclass
class AlgoSeries:
    name: str
    records: List[Record]

# --- DATA LOADING ---
def load_case(path: str) -> Tuple[int, int, List[int], List[int]]:
    # Returns n, capacity, weights, values
    with open(path, "r", encoding="utf-8") as f:
        line1 = f.readline().split()
        if not line1: return 0, 0, [], []
        try:
            n = int(line1[0])
            capacity = int(line1[1])
        except ValueError:
            return 0, 0, [], []

        weights = []
        values = []
        for _ in range(n):
            try:
                parts = list(map(int, f.readline().split()))
                if len(parts) >= 2:
                    weights.append(parts[0])
                    values.append(parts[1])
            except ValueError:
                continue
    return n, capacity, weights, values

def load_ans(path: str) -> Optional[float]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            line = f.readline().strip()
            if not line: return None
            return float(line)
    except (FileNotFoundError, ValueError):
        return None

# --- ALGO WRAPPER ---
def run_astar_wrapper(capacity: int, weights: List[int], values: List[int]) -> Optional[Tuple[float, List[int]]]:
    # Check constraint before running? A* might blow up on huge N
    # Knapsack A* can handle N=40-50 sometimes depending on data
    solver = AStarKnapsack(capacity, weights, values)
    try:
        max_val, selection = solver.solve()
        return max_val, selection
    except Exception as e:
        print(f"A* Error: {e}")
        return None

def run_ga_wrapper(capacity: int, weights: List[int], values: List[int]) -> Optional[Tuple[float, List[int]]]:
    # GA Wrapper
    if GeneticAlgorithmKnapsack is None:
        return None
        
    # Initialize with default params: pop_size=50, generations=100
    try:
        solver = GeneticAlgorithmKnapsack(weights, values, capacity, pop_size=50, generations=100)
        best_ind, best_fitness = solver.run()
        # best_ind is the individual (list of 0/1), best_fitness is the value
        # Make sure best_fitness corresponds to value. In GA implementation:
        # returns self.population[0], history
        # population[0] is (individual, fitness)
        
        # Checking GA implementation again from memory/view_file...
        # view_file says: return self.population[0], history
        # self.population is list of (ind, fitness)
        # so solver.run() returns ((ind, fitness), history)
        
        (ind, fitness), history = solver.run()
        return fitness, ind
    except Exception as e:
        print(f"GA Error: {e}")
        return None

# --- BENCHMARK ---
def bench_series(name: str, runner: Callable[[int, List[int], List[int]], Any], tests_dir: str, num_cases: int) -> AlgoSeries:
    recs: List[Record] = []
    
    try:
        files = [f for f in os.listdir(tests_dir) if f.endswith('.txt')]
        files_map = {}
        for f in files:
            try:
                num = int(os.path.splitext(f)[0])
                files_map[num] = f
            except ValueError:
                pass
        
        sorted_nums = sorted(files_map.keys())
        test_ids = [n for n in sorted_nums if n <= num_cases]
    except FileNotFoundError:
        print(f"Tests directory not found: {tests_dir}")
        return AlgoSeries(name, [])

    for i in test_ids:
        in_path = os.path.join(tests_dir, f"{i}.txt")
        ans_path = os.path.join(tests_dir, f"{i}.ans")
        
        # Check if files exist
        if not os.path.exists(in_path):
            continue

        print(f"  Running Test {i} ({name})...", end="", flush=True)
        
        n, capacity, weights, values = load_case(in_path)
        if n == 0:
             print(" -> Empty/Invalid input")
             continue
             
        expected_val = load_ans(ans_path)
        
        tracemalloc.start()
        start_time = time.perf_counter()
        
        result = runner(capacity, weights, values)
        
        dt = time.perf_counter() - start_time
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak / (1024 * 1024)
        
        got_val = None
        pct_error = 0.0
        
        if result:
            got_val, selection = result
            # Verify validity
            total_w = sum(weights[j] for j in range(n) if selection[j] == 1)
            # Re-sum value to be sure
            total_v = sum(values[j] for j in range(n) if selection[j] == 1)
            
            if total_w > capacity:
                 print(f" -> INVALID! Weight {total_w} > {capacity}", end="")
                 got_val = None
            elif abs(total_v - got_val) > 1e-5:
                 print(f" -> MISMATCH! Calced {total_v} vs Returned {got_val}", end="")
                 got_val = total_v
            
            if got_val is not None and expected_val is not None:
                if expected_val != 0:
                    pct_error = (expected_val - got_val) / expected_val * 100.0 # Deviation from optimal
                else:
                    pct_error = 0.0
            
            print(f" Done. Time: {dt:.4f}s, Val: {got_val} (Exp: {expected_val})")
        else:
            print(f" -> FAILED!", end="")
            print(f" Done. Time: {dt:.4f}s")
            
        recs.append(Record(i, dt, peak_mb, got_val, expected_val, pct_error))
        
    return AlgoSeries(name, recs)

# --- PLOTTING ---
def plot_results(series_list: List[AlgoSeries]):
    if not MATPLOTLIB_AVAILABLE or not SAVE_PLOTS:
        return
        
    os.makedirs(PLOT_DIR, exist_ok=True)
    series_list = [s for s in series_list if s.records]
    if not series_list: return

    # We will create a figure with 2 subplots: Time and Error
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Execution Time
    for s in series_list:
        ids = [r.test_id for r in s.records if r.time_sec is not None]
        times = [r.time_sec for r in s.records if r.time_sec is not None]
        ax1.plot(ids, times, marker="o", label=s.name)
    
    ax1.set_title("Knapsack Execution Time (s)")
    ax1.set_xlabel("Test Case ID")
    ax1.set_ylabel("Time (s)")
    ax1.legend()
    ax1.grid(True)

    # 2. Percentage Error
    # Only if we have expected values
    has_error_data = False
    for s in series_list:
        ids = [r.test_id for r in s.records if r.pct_error is not None]
        errors = [r.pct_error for r in s.records if r.pct_error is not None]
        if ids:
            ax2.plot(ids, errors, marker="x", linestyle="--", label=s.name)
            has_error_data = True
            
    if has_error_data:
        ax2.set_title("Knapsack % Error (Lower is Better)")
        ax2.set_xlabel("Test Case ID")
        ax2.set_ylabel("% Error")
        ax2.legend()
        ax2.grid(True)
    else:
        ax2.set_title("No Expected Values to calc Error")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "kp_comparison.png"), dpi=DPI)
    plt.close()

def main():
    print(f"Starting Knapsack Benchmark...")
    print(f"Test Directory: {TESTS_DIR}")
    
    # Run A*
    print("\n--- Benchmarking A* ---")
    astar_series = bench_series("A*", run_astar_wrapper, TESTS_DIR, NUM_CASES)
    
    # Run GA
    print("\n--- Benchmarking Genetic Algorithm ---")
    ga_series = bench_series("GA", run_ga_wrapper, TESTS_DIR, NUM_CASES)
    
    # Plot
    plot_results([astar_series, ga_series])
    print("\nBenchmark Completed.")

if __name__ == "__main__":
    main()
