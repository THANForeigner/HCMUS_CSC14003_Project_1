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


class ContinuousBenchmark:
    def __init__(self, runs=10, max_iter=100, dim=10, suffix=""):
        self.runs = runs
        self.max_iter = max_iter
        self.dim = dim
        self.suffix = suffix

        folder_name = f"plots{self.suffix}"
        self.plot_dir = os.path.join(BASE_DIR, folder_name)
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
        self.funcs = ["sphere", "rastrigin", "rosenbrock", "griewank", "ackley"]

    def run_hc(self, func_name, func, lb, ub):
        config = algo_config.get("HC", {})
        step_size = 0.05 * (ub - lb) # Dynamic step size taking 5% of bounds
        max_iter = self.max_iter * 30 * self.dim # Equal NFE & Scale with Dimension

        solver = HillClimbing(lb, ub, self.dim, step_size=step_size)
        solver.set_optimization_function(func)
        fit, _ = solver.run(max_iter)
        return fit

    def run_sa(self, func_name, func, lb, ub):
        config = algo_config.get("SA", {})
        initial_temp = config.get("initial_temp", 100.0)
        alpha = config.get("alpha", 0.95)
        final_temp = config.get("final_temp", 0.001)
        max_iter = self.max_iter * self.dim # Scale with Dimension

        solver = SA(bounds=[lb, ub], function=func, dim=self.dim, T=initial_temp, alpha=alpha, stopping_T=final_temp,
                    stopping_iter=max_iter)
        solver.run(times=1)
        return solver.best_result

    def run_de(self, func_name, func, lb, ub):
        config = algo_config.get("DE", {})
        pop_size = config.get("pop_size", 30)
        f = config.get("F", 0.5)
        cr = config.get("CR", 0.9)
        max_iter = self.max_iter * self.dim # Scale with Dimension

        solver = DE(func, bounds=[lb, ub], dim=self.dim, pop_size=pop_size, max_gen=max_iter, F=f, Cr=cr)
        res, _ = solver.run(max_iter)
        return res[0]

    def run_pso(self, func_name, func, lb, ub):
        config = algo_config.get("PSO", {})
        n_particles = config.get("n_particles", 30)
        w = config.get("w", 0.7)
        c1 = config.get("c1", 1.5)
        c2 = config.get("c2", 1.5)
        max_iter = self.max_iter * self.dim # Scale with Dimension

        solver = PSO(function=func, dimension=self.dim, ranges=[lb, ub], swarm_size=n_particles, w=w, c1=c1, c2=c2,
                     max_interation=max_iter)
        solver.run()
        return solver.g_best

    def run_abc(self, func_name, func, lb, ub):
        config = algo_config.get("ABC", {})
        n_bees = config.get("n_bees", 30)
        limit = int((n_bees / 2) * self.dim) # Setup limit dynamically proportional to dimension
        max_iter = self.max_iter * self.dim # Scale with Dimension

        solver = ABC(function=func, ranges=[lb, ub], dimension=self.dim, swarm_size=n_bees, limit=limit,
                     max_iteration=max_iter)
        solver.run()
        return func(solver.best_bee.coords)

    def run_fa(self, func_name, func, lb, ub):
        config = algo_config.get("FA", {})
        n_fireflies = config.get("n_fireflies", 30)
        alpha = config.get("alpha", 0.2)
        beta0 = config.get("beta0", 1.0)
        gamma = 1.0 / np.sqrt(ub - lb) # Dynamic Light absorption
        max_iter = self.max_iter * self.dim # Scale with Dimension

        solver = FA(func_name=func_name, pop_size=n_fireflies, dim=self.dim, max_iter=max_iter, alpha=alpha,
                    beta0=beta0, gamma=gamma)
        _, fit = solver.run()
        return fit

    def run_cs(self, func_name, func, lb, ub):
        config = algo_config.get("CS", {})
        n_nests = config.get("n_nests", 30)
        pa = config.get("pa", 0.25)
        beta = config.get("beta", 1.5)
        max_iter = self.max_iter * self.dim # Scale with Dimension

        solver = CS(func_name=func_name, pop_size=n_nests, dim=self.dim, max_iter=max_iter, pa=pa, beta=beta)
        _, fit = solver.run()
        return fit

    def run_tlbo(self, func_name, func, lb, ub):
        config = algo_config.get("TLBO", {})
        population_size = config.get("population_size", 30)
        max_iter = self.max_iter * self.dim # Scale with Dimension

        solver = TLBO(lb, ub, self.dim, population_size=population_size)
        solver.set_optimization_function(func)
        fit, _ = solver.run(max_iter)
        return fit

    def run(self):
        print("=" * 60)
        print("STARTING CONTINUOUS OPTIMIZATION BENCHMARK")
        print(
            f"Runs per Algo: {self.runs} | Max Iterations: {self.max_iter} | Dimension: {self.dim} | Suffix: '{self.suffix}'")
        print("=" * 60)

        csv_data = []

        for func_name in self.funcs:
            print(f"\n>>> Benchmarking Function: {func_name.upper()} <<<")
            prob = get_problem(func_name)
            func = prob["func"]
            lb = prob["lb"]
            ub = prob["ub"]

            func_results = {}

            for alg_name, wrapper in self.algs.items():
                print(f"  Running {alg_name}...")
                alg_scores = []

                # --- WARM UP PHASE ---
                try:
                    _ = wrapper(func_name, func, lb, ub)
                except:
                    pass
                # ---------------------

                def _run_single_execution():
                    start_t = time.perf_counter()
                    try:
                        score = wrapper(func_name, func, lb, ub)
                    except Exception as e:
                        score = float('inf')
                    end_t = time.perf_counter()
                    return score, end_t - start_t

                # Run parallelly using half the available CPU cores to prevent 100% lockups
                n_workers = max(1, os.cpu_count() // 2)
                results = Parallel(n_jobs=n_workers)(delayed(_run_single_execution)() for _ in range(self.runs))
                
                # Unpack parallel results
                alg_scores = [res[0] for res in results]
                total_time = sum(res[1] for res in results)
                avg_time = total_time / self.runs

                valid_scores = [s for s in alg_scores if s != float('inf')]
                if not valid_scores:
                    print("    -> ALL RUNS FAILED!")
                    continue

                best = np.min(valid_scores)
                worst = np.max(valid_scores)
                mean = np.mean(valid_scores)
                std = np.std(valid_scores)

                print(
                    f"    -> Done. Best: {best:.4e}, Worst: {worst:.4e}, Mean: {mean:.4e}, Std: {std:.4e}, Time: {avg_time:.4f}s")
                func_results[alg_name] = valid_scores

                csv_data.append({
                    "Function": func_name.upper(),
                    "Algorithm": alg_name,
                    "Mean": round(mean, 5),
                    "Std": round(std, 5),
                    "Best": round(best, 5),
                    "Worst": round(worst, 5),
                    "Time (s)": round(avg_time, 5)
                })

            self.draw_boxplot(func_name, func_results)

        self.export_to_csv(csv_data)

    def draw_boxplot(self, func_name, results_dict):
        if not results_dict or not MATPLOTLIB_AVAILABLE:
            return

        labels = list(results_dict.keys())
        data = [results_dict[alg] for alg in labels]

        plt.figure(figsize=(10, 6))
        plt.boxplot(data, tick_labels=labels, patch_artist=True)
        plt.title(f"Performance on {func_name.capitalize()} Function ({self.dim}D, {self.runs} runs)")
        plt.ylabel("Fitness Value (Lower is Better)")
        plt.yscale("log")  # Useful for continuous optimization showing variation of magnitudes
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)

        save_path = os.path.join(self.plot_dir, f"{func_name}_boxplot.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"  * Saved boxplot to {save_path}")

    def export_to_csv(self, data):
        if not data:
            return

        csv_file_name = f"benchmark_results{self.suffix}.csv"
        csv_file_path = os.path.join(BASE_DIR, csv_file_name)

        fieldnames = ["Function", "Algorithm", "Mean", "Std", "Best", "Worst", "Time (s)"]

        with open(csv_file_path, mode='w', newline='', encoding='utf-8-sig') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            writer.writeheader()
            for row in data:
                writer.writerow(row)

        print("=" * 60)
        print(f"[+] Đã xuất dữ liệu báo cáo thành công ra file: {csv_file_path}")
        print("=" * 60)


def performance_test():
    bench_1 = ContinuousBenchmark(runs=30, max_iter=400, dim=20, suffix="_config1")
    start_time_1 = time.time()
    bench_1.run()

    bench_2 = ContinuousBenchmark(runs=30, max_iter=500, dim=30, suffix="_config2")
    start_time_2 = time.time()
    bench_2.run()


