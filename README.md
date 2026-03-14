# Optimization Algorithms Project

This project implements various classical and nature-inspired algorithms to solve several optimization problems, including Knapsack, Shortest Path, Graph Coloring, Traveling Salesman (TSP), and Continuous Function Optimization.

## Project Structure

- `classical/`: Classical search algorithms (Informed, Uninformed, Local Search).
- `nature_inspire/`: Nature-inspired algorithms (Biology-based, Evolution-based, Human-based, Physics-based).
- `problems/`: Problem definitions and input validation utilities.
- `Test/`: Benchmark scripts and test cases for each problem type.
- `visualization/`: GUI-based visualization for continuous optimization algorithms.
- `main.py`: The main entry point with an interactive CLI menu.

## Prerequisites

- Python 3.8+
- Recommended: A virtual environment (`venv`)

### Dependencies

Install the required libraries using `pip`:

```bash
pip install -r requirements.txt
```

## How to Run

### 1. Main Menu (Recommended)
The easiest way to explore the project is through the interactive CLI menu at the root directory. This menu allows you to run benchmarks for all problems and launch the visualization.

```bash
python main.py
```

### 2. Running Benchmarks (Tests)
You can run specific benchmarks either through the `main.py` menu or by executing the `main.py` file within each problem's test directory.

#### Via Main Menu:
Run `python main.py` and select options 1-6.

#### Manually:
Navigate to the specific test directory and run its `main.py`:
- **Knapsack:** `python Test/Knapsack/main.py`
- **Shortest Path:** `python Test/Shortest_Path/main.py`
- **Graph Coloring:** `python Test/Graph_Coloring/main.py`
- **TSP:** `python Test/Traveling_Sale_Man/main.py`
- **Continuous Optimization:** `python Test/Continuous_Optimization/main.py`

### 3. Running Visualization
The visualization tool demonstrates how different nature-inspired algorithms converge on various 3D continuous functions (Sphere, Rastrigin, etc.).

#### Via Main Menu:
Run `python main.py` and select option **7**.

#### Manually:
Run the visualization script directly from the root:
```bash
python visualization/main.py
```

## Supported Algorithms

- **Classical:** A*, BFS, DFS, UCS, Hill Climbing.
- **Nature-Inspired:**
  - **Biology-based:** Particle Swarm Optimization (PSO), Artificial Bee Colony (ABC), Firefly Algorithm (FA), Cuckoo Search (CS), Ant Colony Optimization (ACO).
  - **Evolution-based:** Genetic Algorithm (GA), Differential Evolution (DE).
  - **Human-based:** Teaching-Learning Based Optimization (TLBO).
  - **Physics-based:** Simulated Annealing (SA).
