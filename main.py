from nature_inspire.physic_based.simulated_annealing.travelling_sale_man import SimulatedAnnellingTsp
from nature_inspire.physic_based.simulated_annealing.continuos_functions import SimulatedAnnealingContinuous
from nature_inspire.physic_based.simulated_annealing.graph_coloring import SimulateAnneallingGraphColoring
import matplotlib.pyplot as plt
import random
import nature_inspire.continuous_functions
import numpy as np

def generate_random_graph(num_vertices, density=0.5):
    """
    Generate a random graph.
    density: Probability of an edge between any two vertices (0.0 -> 1.0)
    """
    edges = []
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if random.random() < density:
                edges.append((i, j))
    return edges

def print_graph_info(num_vertices, edges):
    print(f"Graph Generated: {num_vertices} vertices, {len(edges)} edges.")
    print("-" * 40)

def validate_result(edges, solution):
    """Independent verification of the solution"""
    conflicts = 0
    for u, v in edges:
        if solution[u] == solution[v]:
            conflicts += 1
    return conflicts

# ==========================================
# PART 3: MAIN FUNCTION
# ==========================================
if __name__ == "__main__":
    # 1. Configuration
    NUM_VERTICES = 20    # Number of nodes
    MAX_COLORS = 4       # Number of available colors
    DENSITY = 0.4        # Edge density (higher = harder to color)
    
    # 2. Generate Test Data
    print(">>> Generating test data...")
    edges = generate_random_graph(NUM_VERTICES, DENSITY)
    print_graph_info(NUM_VERTICES, edges)

    # 3. Initialize Algorithm
    # T (Temperature): Higher helps escape local optima
    # Alpha: Cooling rate (0.995 = slow/thorough, 0.90 = fast/greedy)
    sa = SimulateAnneallingGraphColoring(
        max_colors=MAX_COLORS, 
        max_vertices=NUM_VERTICES, 
        edges=edges, 
        T=10.0, 
        alpha=0.995,
        stopping_iter=50000
    )

    # 4. Run Algorithm
    # We run it multiple times to avoid bad random starting points
    sa.batch_annealling(times=5)

    # 5. Print Final Results
    print("\n>>> FINAL RESULTS:")
    print(f"Lowest Energy (conflicting edges): {sa.best_energy}")
    print(f"Color Assignment: {sa.best_solution}")

    # 6. Double Check
    real_conflicts = validate_result(edges, sa.best_solution)
    if real_conflicts == 0:
        print("\n SUCCESS: Valid graph coloring found!")
    else:
        print(f"\n FAILURE: {real_conflicts} edges still have conflicts.")
        print("Tip: Try increasing the number of colors or the stopping iterations.")