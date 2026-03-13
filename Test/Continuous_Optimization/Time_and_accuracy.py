import os
import sys
import numpy as np
import time
import csv
import os
from joblib import Parallel, delayed

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not found. Plotting disabled.")

# Ensure root paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
sys.path.append(PROJECT_ROOT)

from classical.local.hill_climbing import HillClimbing
from nature_inspire.physic_based.simulated_annealing.simulated_annealing import SA
from nature_inspire.evolution_based.differential_evolution.differential_evolution import DE
from nature_inspire.biology_based.particle_swarm_optimization.particle_swarm_optimization import PSO
from nature_inspire.biology_based.artificial_bee_colony.artificial_bee_colony import ABC
from nature_inspire.biology_based.firefly_algorithm.firefly_algorithm import FA
from nature_inspire.biology_based.cuckoo_search.cuckoo_search import CS
from nature_inspire.human_based.teaching_learning_based_optimization.teaching_learning_based_optimization import TLBO
from problems.problem import get_problem, algo_config
try:
    from problems.input_validator import CONT_FUNCS, CONT_RUNS
except ImportError:
    CONT_FUNCS = ["sphere", "rastrigin", "rosenbrock", "griewank", "ackley"]
    CONT_RUNS = 10


class DimensionalityBenchmark:
    def __init__(self, runs=CONT_RUNS, max_iter=500, max_dim=20, tolerance=1e-4):
        self.runs = runs
        self.max_iter = max_iter
        self.max_dim = max_dim
        self.tolerance = tolerance  # Ngưỡng fitness để coi là tìm được nghiệm "đúng"
        
        self.plot_dir = os.path.join(BASE_DIR, "plots")
        os.makedirs(self.plot_dir, exist_ok=True)

        self.algs = {
            "HC": self.run_hc,
            "SA": self.run_sa,
            "DE": self.run_de,
            "PSO": self.run_pso,
            "ABC": self.run_abc,
            "FA": self.run_fa,
            "CS": self.run_cs,
            "TLBO": self.run_tlbo
        }
        self.funcs = CONT_FUNCS

    def run_hc(self, func_name, func, lb, ub, dim):
        config = algo_config.get("HC", {})
        step_size = config.get("step_size", 0.5)
        max_iter = self.max_iter

        solver = HillClimbing(lb, ub, dim, step_size=step_size)
        solver.set_optimization_function(func)
        fit, _ = solver.run(max_iter)
        return fit

    def run_sa(self, func_name, func, lb, ub, dim):
        config = algo_config.get("SA", {})
        initial_temp = config.get("initial_temp", 1000.0)
        alpha = config.get("alpha", 0.95)
        final_temp = config.get("final_temp", 1e-8)
        max_iter = self.max_iter * dim # Scale with Dimension

        solver = SA(bounds=[lb, ub], function=func, dim=dim, T=initial_temp, step_size=0.5, alpha=alpha, stopping_T=final_temp, stopping_iter=max_iter)
        solver.run()
        return solver.best_result

    def run_de(self, func_name, func, lb, ub, dim):
        config = algo_config.get("DE", {})
        pop_size = config.get("pop_size", 30)
        max_iter = self.max_iter

        solver = DE(func, bounds=[lb, ub], dim=dim, pop_size=pop_size, max_gen=max_iter)
        res, _ = solver.run(max_iter)
        return res[0]

    def run_pso(self, func_name, func, lb, ub, dim):
        config = algo_config.get("PSO", {})
        n_particles = config.get("n_particles", 30)
        w = config.get("w", 0.7)
        c1 = config.get("c1", 1.5)
        c2 = config.get("c2", 1.5)
        max_iter = self.max_iter * dim # Scale with Dimension

        solver = PSO(function=func, dimension=dim, ranges=[lb, ub], swarm_size=n_particles, w=w, c1=c1, c2=c2, max_interation=max_iter)
        solver.run()
        return solver.g_best

    def run_abc(self, func_name, func, lb, ub, dim):
        config = algo_config.get("ABC", {})
        n_bees = config.get("n_bees", 30)
        limit = int((n_bees / 2) * dim) # Dynamic limit
        max_iter = self.max_iter * dim # Scale with Dimension

        solver = ABC(function=func, ranges=[lb, ub], dimension=dim, swarm_size=n_bees, limit=limit, max_iteration=max_iter)
        solver.run()
        return func(solver.best_bee.coords)

    def run_fa(self, func_name, func, lb, ub, dim):
        config = algo_config.get("FA", {})
        n_fireflies = config.get("n_fireflies", 30)
        alpha = config.get("alpha", 0.2)
        beta0 = config.get("beta0", 1.0)
        gamma = 1.0 / np.sqrt(ub - lb) # Dynamic gamma
        max_iter = self.max_iter * dim # Scale with Dimension

        solver = FA(func_name=func_name, pop_size=n_fireflies, dim=dim, max_iter=max_iter, alpha=alpha, beta0=beta0, gamma=gamma)
        _, fit = solver.run()
        return fit

    def run_cs(self, func_name, func, lb, ub, dim):
        config = algo_config.get("CS", {})
        n_nests = config.get("n_nests", 30)
        pa = config.get("pa", 0.25)
        beta = config.get("beta", 1.5)
        max_iter = self.max_iter * dim # Scale with Dimension

        solver = CS(func_name=func_name, pop_size=n_nests, dim=dim, max_iter=max_iter, pa=pa, beta=beta)
        _, fit = solver.run()
        return fit

    def run_tlbo(self, func_name, func, lb, ub, dim):
        config = algo_config.get("TLBO", {})
        population_size = config.get("population_size", 30)
        max_iter = self.max_iter * dim # Scale with Dimension

        solver = TLBO(lb, ub, dim, population_size=population_size)
        solver.set_optimization_function(func)
        fit, _ = solver.run(max_iter)
        return fit

    def run(self):
        print("=" * 70)
        print("STARTING DIMENSIONALITY SCALABILITY BENCHMARK")
        print(f"Runs per Dim: {self.runs} | Max Iter: {self.max_iter} | Max Dim: {self.max_dim}")
        print(f"Success Tolerance: < {self.tolerance}")
        print("=" * 70)

        csv_data = []
        dimensions = list(range(1, self.max_dim + 1))

        for func_name in self.funcs:
            print(f"\n>>> Benchmarking Function: {func_name.upper()} <<<")
            prob = get_problem(func_name)
            func = prob["func"]
            lb = prob["lb"]
            ub = prob["ub"]

            # Dữ liệu phục vụ vẽ biểu đồ
            time_plot_data = {alg: [] for alg in self.algs}
            success_plot_data = {alg: [] for alg in self.algs}

            for alg_name, wrapper in self.algs.items():
                print(f"  Running {alg_name}...")
                
                # --- WARM UP PHASE ---
                # Chạy nháp 1 lần với 1D để loại bỏ "Cold start" (thời gian nạp thư viện/JIT) 
                try:
                    _ = wrapper(func_name, func, lb, ub, 1)
                except:
                    pass
                # ---------------------
                
                for dim in dimensions:
                    def _run_single_execution():
                        start_t = time.perf_counter()
                        try:
                            score = wrapper(func_name, func, lb, ub, dim)
                            is_success = not np.isnan(score) and score < (self.tolerance * dim)
                        except Exception:
                            is_success = False
                        end_t = time.perf_counter()
                        return is_success, end_t - start_t

                    # Execute all 30 runs for this dimension in parallel (limit to half cores to prevent lockup)
                    # Execute all 30 runs for this dimension in parallel (limit to half cores to prevent lockup)
                    print(f"      Dim {dim}: ", end="", flush=True)
                    dim_start_time = time.time()
                    n_workers = max(1, os.cpu_count() // 4)
                    results = Parallel(n_jobs=n_workers)(delayed(_run_single_execution)() for _ in range(self.runs))
                    dim_time_taken = time.time() - dim_start_time
                    mins, secs = divmod(dim_time_taken, 60)
                    if mins > 0:
                        print(f"Done in {int(mins)}m {secs:.2f}s")
                    else:
                        print(f"Done in {secs:.2f}s")

                    # Aggregate results
                    success_count = sum(1 for res in results if res[0])
                    total_time = sum(res[1] for res in results)

                    avg_time = total_time / self.runs
                    success_rate = (success_count / self.runs) * 100  # Tính bằng %

                    time_plot_data[alg_name].append(avg_time)
                    success_plot_data[alg_name].append(success_rate)

                    # Lưu vào dữ liệu CSV
                    csv_data.append({
                        "Function": func_name.upper(),
                        "Algorithm": alg_name,
                        "Dimension": dim,
                        "Success Rate (%)": round(success_rate, 2),
                        "Avg Time (s)": round(avg_time, 5)
                    })
                    
                print(f"    -> Done for dims 1 to {self.max_dim}.")

            # Vẽ Line Chart cho hàm hiện tại
            self.draw_line_charts(func_name, dimensions, time_plot_data, success_plot_data)

        # Xuất file CSV
        self.export_to_csv(csv_data)

    def draw_line_charts(self, func_name, dimensions, time_data, success_data):
        if not MATPLOTLIB_AVAILABLE:
            return

        # 1. Vẽ biểu đồ thời gian chạy (Execution Time)
        plt.figure(figsize=(12, 6))
        for alg, times in time_data.items():
            plt.plot(dimensions, times, marker='o', markersize=4, label=alg)
            
        plt.title(f"Execution Time vs Dimension on {func_name.capitalize()} Function")
        plt.xlabel("Dimension (1 to 20)")
        plt.ylabel("Average Time (seconds)")
        plt.xticks(dimensions)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        time_save_path = os.path.join(self.plot_dir, f"{func_name}_time_linechart.png")
        plt.savefig(time_save_path, dpi=200)
        plt.close()
        print(f"  * Saved Time chart to {time_save_path}")

        # 2. Vẽ biểu đồ tỷ lệ đúng (Success Rate)
        plt.figure(figsize=(12, 6))
        for alg, rates in success_data.items():
            plt.plot(dimensions, rates, marker='s', markersize=4, label=alg)
            
        plt.title(f"Success Rate vs Dimension on {func_name.capitalize()} Function")
        plt.xlabel("Dimension (1 to 20)")
        plt.ylabel("Success Rate (%)")
        plt.xticks(dimensions)
        plt.ylim(-5, 105) # Rate từ 0 đến 100%
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        success_save_path = os.path.join(self.plot_dir, f"{func_name}_success_linechart.png")
        plt.savefig(success_save_path, dpi=200)
        plt.close()
        print(f"  * Saved Success Rate chart to {success_save_path}")

    def export_to_csv(self, data):
        if not data:
            return

        csv_file_path = os.path.join(BASE_DIR, "scalability_benchmark_results.csv")
        fieldnames = ["Function", "Algorithm", "Dimension", "Success Rate (%)", "Avg Time (s)"]

        with open(csv_file_path, mode='w', newline='', encoding='utf-8-sig') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow(row)

        print("=" * 70)
        print(f"[+] Đã xuất báo cáo thành công ra file: {csv_file_path}")
        print("=" * 70)

def time_and_accuracy_test():
    # 10 lần chạy, max iter tùy bạn chỉnh, max_dim = 20
    benchmark = DimensionalityBenchmark(runs=30, max_iter=1000, max_dim=20, tolerance=0.1)
    benchmark.run()
