import sys
import os
from contextlib import redirect_stdout, contextmanager

from Test.Knapsack.main import KnapsackBenchmark
from Test.Shortest_Path.main import ShortestPathBenchmark
from Test.Graph_Coloring.main import GraphColoringBenchmark
from Test.Traveling_Sale_Man.main import TSPBenchmark
from Test.Continuous_Optimization.main import ContinuousBenchmark

try:
    from problems.problem import algo_config
except ImportError:
    algo_config = {}

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
@contextmanager
def set_working_dir(directory):
    original_dir = os.getcwd()
    try:
        os.chdir(directory)
        yield
    finally:
        os.chdir(original_dir)


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

class SingleLineStream:
    def __init__(self):
        self.terminal = sys.__stdout__
        self.last_len = 0

    def write(self, text):
        clean_text = text.replace('\n', '').replace('\r', '').strip()
        if clean_text:
            display_text = clean_text[:100]
            out = '\r>> Progress: ' + display_text.ljust(self.last_len)
            self.terminal.write(out)
            self.terminal.flush()
            self.last_len = len(out) - 14

    def flush(self):
        pass


def run_with_progress(benchmark_func):
    """Hàm bọc để chạy thuật toán và gom output vào 1 dòng"""
    with redirect_stdout(SingleLineStream()):
        benchmark_func()
    print("\n[✔] HOÀN THÀNH!")


def run_knapsack():
    print("\n" + "=" * 50)
    print("RUNNING KNAPSACK BENCHMARK...")
    target_dir = os.path.join(ROOT_DIR, "Test", "Knapsack")
    with set_working_dir(target_dir):
        benchmark = KnapsackBenchmark()
        run_with_progress(benchmark.run)


def run_shortest_path():
    print("\n" + "=" * 50)
    print("RUNNING SHORTEST PATH BENCHMARK...")
    target_dir = os.path.join(ROOT_DIR, "Test", "Shortest_Path")
    with set_working_dir(target_dir):
        benchmark = ShortestPathBenchmark()
        run_with_progress(benchmark.run)


def run_graph_coloring():
    print("\n" + "=" * 50)
    print("RUNNING GRAPH COLORING BENCHMARK...")
    target_dir = os.path.join(ROOT_DIR, "Test", "Graph_Coloring")
    with set_working_dir(target_dir):
        benchmark = GraphColoringBenchmark()
        run_with_progress(benchmark.run)


def run_tsp():
    print("\n" + "=" * 50)
    print("RUNNING TRAVELING SALESMAN (TSP) BENCHMARK...")
    target_dir = os.path.join(ROOT_DIR, "Test", "Traveling_Sale_Man")
    with set_working_dir(target_dir):
        benchmark = TSPBenchmark()
        run_with_progress(benchmark.run)


def run_continuous_optimization():
    print("\n" + "=" * 50)
    print("RUNNING CONTINUOUS OPTIMIZATION BENCHMARK...")
    target_dir = os.path.join(ROOT_DIR, "Test", "Continuous_Optimization")
    with set_working_dir(target_dir):
        config = algo_config.get("Continuous_Optimization", {})
        runs = config.get("runs", 10)
        max_iter = config.get("max_iter", 100)
        dim = config.get("dim", 10)

        benchmark = ContinuousBenchmark(runs=runs, max_iter=max_iter, dim=dim)
        run_with_progress(benchmark.run)


def run_all():
    print("\n" + "*" * 50)
    print("RUNNING ALL BENCHMARKS SEQUENTIALLY")
    print("*" * 50)
    run_knapsack()
    run_shortest_path()
    run_graph_coloring()
    run_tsp()
    run_continuous_optimization()
    print("\n" + "*" * 50)
    print("ALL BENCHMARKS COMPLETED")
    print("*" * 50)


def wait_for_any_key():
    print("\nPress ANY KEY to return to menu...", end='', flush=True)
    if os.name == 'nt':
        import msvcrt
        msvcrt.getch()
    else:
        try:
            import tty
            import termios
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            input("\nPress Enter to return to menu...")
    print()

def show_menu():
    print("            ALGORITHM BENCHMARK MENU")
    print("=" * 50)
    print("1. Run Knapsack Benchmark")
    print("2. Run Shortest Path Benchmark")
    print("3. Run Graph Coloring Benchmark")
    print("4. Run Traveling Salesman Benchmark")
    print("5. Run Continuous Optimization Benchmark")
    print("6. Run All Benchmarks Sequence")
    print("0. Exit")
    print("=" * 50)


def main():
    while True:
        clear_screen()
        show_menu()

        try:
            choice = input("Enter your choice (0-6): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            sys.exit(0)

        if choice == '1':
            run_knapsack()
        elif choice == '2':
            run_shortest_path()
        elif choice == '3':
            run_graph_coloring()
        elif choice == '4':
            run_tsp()
        elif choice == '5':
            run_continuous_optimization()
        elif choice == '6':
            run_all()
        elif choice == '0':
            clear_screen()
            print("Exiting...")
            sys.exit(0)
        else:
            print("Invalid choice. Please enter a number between 0 and 6.")

        if choice != '0':
            input("\nPress Enter to return to menu...")


if __name__ == "__main__":
    main()