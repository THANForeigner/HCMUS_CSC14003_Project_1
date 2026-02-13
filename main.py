import nature_inspire.physic_based.simulated_annealing as sa
import nature_inspire.biology_based.particle_swarm_optimization.continous_functions as pso
import nature_inspire.biology_based.artificial_bee_colony.continuous_functions as abc
import nature_inspire.biology_based.artificial_bee_colony.knapsack as abck
import matplotlib.pyplot as plt
import random
import nature_inspire.continuous_functions as cf
import numpy as np

if __name__ == "__main__":
    # Ví dụ dữ liệu test
    weights = [10, 20, 30, 40, 50]
    values = [20, 30, 66, 40, 60]
    capacity = 100
    n_items = 5
    
    abc = abck.ArtificialBeeColonyKnapsack(
        items=n_items, 
        capacity=capacity, 
        weight=weights, 
        cost=values, 
        swarm_size=20, 
        limit=5, 
        max_iteration=50
    )
    abc.artificial_bee_colony()
    best_fitness = abc.best_bee.fitness
    best_coords = abc.best_bee.coords
    print("Best Value:", best_fitness)
    print("Selected Items:", best_coords)