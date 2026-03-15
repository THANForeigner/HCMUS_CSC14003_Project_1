from __future__ import annotations
import os
import sys
import signal
import time
import tracemalloc
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Any

# Ensure project root and necessary module paths are in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'nature_inspire', 'evolution_based', 'genetic_algorithm'))

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
    from nature_inspire.evolution_based.genetic_algorithm.genetic_algorithm_knapsack import GA_Knapsack as GeneticAlgorithmKnapsack
except ImportError as e:
    print(f"Warning: GeneticAlgorithmKnapsack (GA_Knapsack) not found. {e}")

try:
    from nature_inspire.biology_based.artificial_bee_colony.artificial_bee_colony_knapsack import ABC_Knapsack
except ImportError as e:
    ABC_Knapsack = None
    print(f"Warning: ABC_Knapsack not found. {e}")


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
    from problems.input_validator import load_knapsack_cases, KNAPSACK_DIR
except ImportError as _e:
    print(f"Warning: input_validator not found ({_e}).")
    load_knapsack_cases = None; KNAPSACK_DIR = None

# --- TIMEOUT HANDLER ---
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = KNAPSACK_DIR or os.path.join(BASE_DIR, "tests_knapsack")
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
    solver = AStarKnapsack(capacity, weights, values)
    try:
        max_val, selection = solver.run()
        return max_val, selection
    except Exception as e:
        print(f"A* Error: {e}")
        return None

def run_ga_wrapper(capacity: int, weights: List[int], values: List[int]) -> Optional[Tuple[float, List[int]]]:
    if GeneticAlgorithmKnapsack is None:
        return None
        
    ga_config = algo_config.get("GA", {})
    pop_size = ga_config.get("population_size", 50)
    generations = ga_config.get("max_iter", 100)
    crossover_rate = ga_config.get("crossover_rate", 0.9)
    mutation_rate = ga_config.get("mutation_rate", 0.1)
    elite_size = ga_config.get("elite_size", 2)
    
    try:
        solver = GeneticAlgorithmKnapsack(
            weights=weights, 
            values=values, 
            capacity=capacity, 
            pop_size=pop_size, 
            generations=generations,
            mutation_rate=mutation_rate
        )
        # GA.run() returns (best_individual_tuple, history)
        # best_individual_tuple = (ind, fitness)
        (ind, fitness), history = solver.run()
        return fitness, ind
    except Exception as e:
        print(f"GA Error: {e}")
        return None

def run_abc_wrapper(capacity: int, weights: List[int], values: List[int]) -> Optional[Tuple[float, List[int]]]:
    if ABC_Knapsack is None:
        return None
    
    n = len(weights)
    abc_config = algo_config.get("ABC", {})
    # Correct key is 'n_bees', not 'swarm_size'
    swarm_size = abc_config.get("n_bees", 40)
    max_iter = abc_config.get("max_iter", 100)
    
    # Scale max_iteration for discrete ABC to ensure it runs more than 0 seconds on large tests
    max_iter = max(max_iter, n * 10)
    
    limit = abc_config.get("limit", 20)
    
    try:
        solver = ABC_Knapsack(
            items=n, 
            capacity=capacity, 
            weight=weights, 
            cost=values, 
            swarm_size=swarm_size, 
            limit=limit, 
            max_iteration=max_iter
        )
        solver.run()
        return solver.best_bee.fitness, solver.best_bee.coords.tolist()
    except Exception as e:
        print(f"ABC Error: {e}")
        return None

# --- BENCHMARK CLASS ---
class KnapsackBenchmark:
    def __init__(self, tests_dir: str = TESTS_DIR, plot_dir: str = PLOT_DIR, num_cases: int = NUM_CASES, save_plots: bool = SAVE_PLOTS, dpi: int = DPI):
        self.tests_dir = tests_dir
        self.plot_dir = plot_dir
        self.num_cases = num_cases
        self.save_plots = save_plots
        self.dpi = dpi

    def bench_series(self, name: str, runner: Callable[[int, List[int], List[int]], Any]) -> AlgoSeries:
        recs: List[Record] = []

        # Dùng load_knapsack_cases() từ input_validator thay vì tự load
        if load_knapsack_cases:
            cases = load_knapsack_cases(tests_dir=self.tests_dir, num=self.num_cases)
        else:
            cases = []  # Fallback: không có loader

        for case in cases:
            i, capacity, weights, values, expected_val = case
            n = len(weights)
            print(f"  Running Test {i} ({name})...", end="", flush=True)
            
            tracemalloc.start()
            start_time = time.perf_counter()
            
            result = None
            
            # Setup timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)  # 1 minute timeout
            
            try:
                result = runner(capacity, weights, values)
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

    def plot_results(self, series_list: List[AlgoSeries]):
        if not MATPLOTLIB_AVAILABLE or not self.save_plots:
            return
            
        os.makedirs(self.plot_dir, exist_ok=True)
        series_list = [s for s in series_list if s.records]
        if not series_list: return

        # We will create a figure with 2 subplots: Time and Error
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 1. Execution Time
        markers = ['o', '*', 's', '^', 'D', 'v', 'p', 'h']
        for i, s in enumerate(series_list):
            m = markers[i % len(markers)]
            ids = [r.test_id for r in s.records if r.time_sec is not None]
            times = [r.time_sec for r in s.records if r.time_sec is not None]
            ax1.plot(ids, times, marker=m, linestyle="dashed", label=s.name)
        
        ax1.set_title("Knapsack Problem - Execution Time (s)")
        ax1.set_xlabel("Test Case ID")
        ax1.set_ylabel("Time (s)")
        ax1.legend()
        ax1.grid(True)

        # 2. Percentage Error
        # Only if we have expected values
        has_error_data = False
        markers = ['o', '*', 's', '^', 'D', 'v', 'p', 'h']
        for i, s in enumerate(series_list):
            m = markers[i % len(markers)]
            ids = [r.test_id for r in s.records if r.pct_error is not None]
            errors = [r.pct_error for r in s.records if r.pct_error is not None]
            if ids:
                ax2.plot(ids, errors, marker=m, linestyle="--", label=s.name)
                has_error_data = True
                
        if has_error_data:
            ax2.set_title("Knapsack Problem - % Error (Lower is Better)")
            ax2.set_xlabel("Test Case ID")
            ax2.set_ylabel("% Error")
            ax2.legend()
            ax2.grid(True)
        else:
            ax2.set_title("No Expected Values to calc Error")

        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "kp_comparison.png"), dpi=self.dpi)
        plt.close()

    def run(self):
        print(f"Starting Knapsack Benchmark...")
        print(f"Test Directory: {self.tests_dir}")
        
        # --- WARM-UP: run each algorithm on a trivial input to preload libraries/JIT ---
        print("Warming up algorithms (eliminating cold start)...", end="", flush=True)
        import contextlib, io
        _dummy_cap, _dummy_w, _dummy_v = 10, [2, 3], [4, 5]
        with contextlib.redirect_stdout(io.StringIO()):
            try: run_astar_wrapper(_dummy_cap, _dummy_w, _dummy_v)
            except Exception: pass
            try: run_ga_wrapper(_dummy_cap, _dummy_w, _dummy_v)
            except Exception: pass
            try: run_abc_wrapper(_dummy_cap, _dummy_w, _dummy_v)
            except Exception: pass
        print(" Done.")
        
        # Run A*
        print("\n--- Benchmarking A* ---")
        astar_series = self.bench_series("A*", run_astar_wrapper)
        
        # Run GA
        print("\n--- Benchmarking Genetic Algorithm ---")
        ga_series = self.bench_series("GA", run_ga_wrapper)
        
        # Run ABC
        print("\n--- Benchmarking Artificial Bee Colony ---")
        abc_series = self.bench_series("ABC", run_abc_wrapper)
        
        # Plot
        self.plot_results([astar_series, ga_series, abc_series])
        print("\nBenchmark Completed.")

def main():
    benchmark = KnapsackBenchmark()
    benchmark.run()

if __name__ == "__main__":
    main()
