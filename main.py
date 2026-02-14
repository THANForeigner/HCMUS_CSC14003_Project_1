import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import 3D toolkit
import nature_inspire.biology_based.particle_swarm_optimization.continous_functions as pso
import nature_inspire.continuous_functions as cf

if __name__ == "__main__":
    DIMENSION = 2
    BOUNDS = [-5.12, 5.12]
    SWARM_SIZE = 50
    MAX_ITER = 100

    # Define function
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

    # --- 3D PLOTTING SECTION ---
    if DIMENSION == 2:
        print("\nPlotting results in 3D...")

        # 1. Create data for the 3D surface
        x = np.linspace(BOUNDS[0], BOUNDS[1], 100)
        y = np.linspace(BOUNDS[0], BOUNDS[1], 100)
        X, Y = np.meshgrid(x, y)
        
        # Calculate Z for every point on the meshgrid
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = function([X[i, j], Y[i, j]])

        # 2. Setup the 3D Figure
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 3. Plot the Surface
        # alpha=0.6 makes it semi-transparent so you can see points 'inside' or behind it
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.6)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Fitness Value')

        # 4. Plot the True Global Minimum (at 0,0,0)
        ax.scatter(0, 0, 0, color='red', marker='x', s=100, linewidth=3, label='True Global Min (0,0)', zorder=10)

        # 5. Plot the PSO Result
        # We need 3 coordinates: x, y, and z (fitness)
        found_x = optimizer.g_best_pos[0]
        found_y = optimizer.g_best_pos[1]
        found_z = optimizer.g_best
        
        ax.scatter(found_x, found_y, found_z, color='black', marker='o', s=100, label='PSO Result', zorder=10)

        # Labels and View
        ax.set_title(f'3D PSO Result on Sphere Function\nBest Fitness: {optimizer.g_best:.6f}')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Fitness (Z)')
        
        # Optional: Set initial viewing angle (elevation, azimuth)
        ax.view_init(elev=30, azim=45)

        plt.legend()
        plt.show()