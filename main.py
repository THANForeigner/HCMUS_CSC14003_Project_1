import sys
import os
import subprocess
from collections import deque
from rich.live import Live
from rich.panel import Panel
from rich.console import Group
from rich.text import Text

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def run_benchmark_with_pin(title, target_dir):
    MAX_LINES = 25  # Số dòng output tối đa hiển thị
    log_lines = deque(maxlen=MAX_LINES)
    custom_env = os.environ.copy()
    if "PYTHONPATH" in custom_env:
        custom_env["PYTHONPATH"] = f"{ROOT_DIR}{os.pathsep}{custom_env['PYTHONPATH']}"
    else:
        custom_env["PYTHONPATH"] = ROOT_DIR

    # Khởi chạy file main.py tại thư mục đích với custom_env
    process = subprocess.Popen([sys.executable, "-u", "main.py"],
        cwd=target_dir,
        env=custom_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace'
    )

    def get_renderable():
        return Group(
            Panel(f"[bold green]{title}[/bold green]", border_style="cyan", expand=False),
            Text("\n".join(log_lines))
        )

    try:
        with Live(get_renderable(), refresh_per_second=15) as live:
            for line in iter(process.stdout.readline, ''):
                if line:
                    log_lines.append(line.rstrip('\r\n'))
                    live.update(get_renderable())
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        print("\n[Đã dừng thuật toán]")

def run_knapsack():
    target_dir = os.path.join(ROOT_DIR, "Test", "Knapsack")
    run_benchmark_with_pin("📌 RUNNING KNAPSACK BENCHMARK...", target_dir)


def run_shortest_path():
    target_dir = os.path.join(ROOT_DIR, "Test", "Shortest_Path")
    run_benchmark_with_pin("📌 RUNNING SHORTEST PATH BENCHMARK...", target_dir)


def run_graph_coloring():
    target_dir = os.path.join(ROOT_DIR, "Test", "Graph_Coloring")
    run_benchmark_with_pin("📌 RUNNING GRAPH COLORING BENCHMARK...", target_dir)


def run_tsp():
    target_dir = os.path.join(ROOT_DIR, "Test", "Traveling_Sale_Man")
    run_benchmark_with_pin("📌 RUNNING TRAVELING SALESMAN (TSP) BENCHMARK...", target_dir)


def run_continuous_optimization():
    target_dir = os.path.join(ROOT_DIR, "Test", "Continuous_Optimization")
    run_benchmark_with_pin("📌 RUNNING CONTINUOUS OPTIMIZATION BENCHMARK...", target_dir)


def run_visualization():
    target_dir = os.path.join(ROOT_DIR, "visualization")
    print(f"🎨 RUNNING ALGORITHM VISUALIZATION in {target_dir}...")
    
    custom_env = os.environ.copy()
    if "PYTHONPATH" in custom_env:
        custom_env["PYTHONPATH"] = f"{ROOT_DIR}{os.pathsep}{custom_env['PYTHONPATH']}"
    else:
        custom_env["PYTHONPATH"] = ROOT_DIR

    try:
        # Run directly without standard output pinning (Rich) to avoid interference with GUI/Animation
        subprocess.run([sys.executable, "main.py"], cwd=target_dir, env=custom_env)
    except KeyboardInterrupt:
        print("\n[Đã dừng Visualization]")


def run_all_discrete():
    print("\n" + "*" * 50)
    print("🚀 RUNNING ALL DISCRETE BENCHMARKS SEQUENTIALLY")
    print("*" * 50)

    run_knapsack()
    run_shortest_path()
    run_graph_coloring()
    run_tsp()

    print("\n" + "*" * 50)
    print("✅ ALL DISCRETE BENCHMARKS COMPLETED")
    print("*" * 50)


def show_menu():
    print("            ALGORITHM BENCHMARK MENU")
    print("=" * 50)
    print("1. Run Knapsack Benchmark")
    print("2. Run Shortest Path Benchmark")
    print("3. Run Graph Coloring Benchmark")
    print("4. Run Traveling Salesman Benchmark")
    print("5. Run Continuous Optimization Benchmark")
    print("6. Run All Discrete Benchmarks (1-4)")
    print("7. Run Algorithm Visualization")
    print("0. Exit")
    print("=" * 50)


def main():
    while True:
        clear_screen()
        show_menu()

        try:
            choice = input("Enter your choice (0-7): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            sys.exit(0)

        # Xóa màn hình trước khi chạy thuật toán
        clear_screen()

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
            run_all_discrete()
        elif choice == '7':
            run_visualization()
        elif choice == '0':
            print("Exiting...")
            sys.exit(0)
        else:
            print("Invalid choice. Please enter a number between 0 and 7.")

        if choice != '0':
            input("\nPress Enter to return to menu...")


if __name__ == "__main__":
    main()