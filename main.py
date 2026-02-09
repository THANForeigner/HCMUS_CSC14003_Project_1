import nature_inspire.physic_based.simulated_annealing as sa
import nature_inspire.biology_based.particle_swarm_optimization.continous_functions as pso
import matplotlib.pyplot as plt
import random
import nature_inspire.continuous_functions as cf
import numpy as np

if __name__ == "__main__":
    DIMENSION = 2           
    BOUNDS = [-5.12, 5.12]  
    SWARM_SIZE = 50         
    MAX_ITER = 100          
    function = cf.rastrigin_function
    print(f"--- Starting PSO test with Sphere function ({DIMENSION} dimensions) ---")
    optimizer = pso.ParticleSwarmOptimization(
        function=function,
        dimension=DIMENSION,
        ranges=BOUNDS,
        swarm_size=SWARM_SIZE,
        max_interation=MAX_ITER  
    )
    optimizer.particle_swarm_optimization()
    print("\n--- RESULTS ---")
    print(f"Global Best Fitness: {optimizer.g_best:.10f}")
    best_pos_rounded = np.round(optimizer.g_best_pos, 5)
    print(f"Global Best Position: {best_pos_rounded}")
    if DIMENSION == 2:
        print("\nDrawing...")
        x = np.linspace(BOUNDS[0], BOUNDS[1], 100)
        y = np.linspace(BOUNDS[0], BOUNDS[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = function([X[i, j], Y[i, j]])

        plt.figure(figsize=(10, 8))
        plt.contourf(X, Y, Z, levels=50, cmap='viridis')
        plt.colorbar(label='Fitness Value')

        plt.scatter(0, 0, color='white', marker='x', s=100, label='True Global Min (0,0)')

        found_x = optimizer.g_best_pos[0]
        found_y = optimizer.g_best_pos[1]
        plt.scatter(found_x, found_y, color='red', marker='o', s=100, edgecolors='black', label='PSO Result')

        plt.title(f'PSO Result on Sphere Function\nBest Fitness: {optimizer.g_best:.6f}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.show()