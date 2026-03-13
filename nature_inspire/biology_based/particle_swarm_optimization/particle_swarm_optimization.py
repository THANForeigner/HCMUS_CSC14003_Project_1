import numpy as np
import random


class PSO:
    def __init__(self, function, dimension=-1, ranges=None, swarm_size=-1,
                 w=-1, c1=-1, c2=-1, max_interation=-1):
        self.function = function
        self.dim = 2 if dimension == -1 else dimension
        self.ranges = ranges
        self.swarm_size = 50 if swarm_size == -1 else swarm_size

        # Key coefficients for the PSO algorithm
        self.w = 0.729 if w == -1 else w  # Inertia weight
        self.c1 = 1.49445 if c1 == -1 else c1  # Cognitive coefficient
        self.c2 = 1.49445 if c2 == -1 else c2  # Social coefficient

        self.max_iter = 1000 if max_interation == -1 else max_interation
        self.iter = 0
        self.history = []

        # Vectorized initialization of positions and velocities
        self.pos = np.random.uniform(self.ranges[0], self.ranges[1], (self.swarm_size, self.dim))
        self.velocity = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        
        self.p_best_pos = self.pos.copy()
        
        # Calculate initial fitness for the entire swarm concurrently
        self.p_best = np.apply_along_axis(self.function, 1, self.pos)
        
        # Determine global best
        best_idx = np.argmin(self.p_best)
        self.g_best = self.p_best[best_idx]
        self.g_best_pos = self.p_best_pos[best_idx].copy()

    def run(self):
        self.history.append(self.pos.copy().tolist())

        while self.iter < self.max_iter:
            # Generate random coefficients for the entire swarm
            r1 = np.random.rand(self.swarm_size, self.dim)
            r2 = np.random.rand(self.swarm_size, self.dim)

            # Calculate velocity components
            inertia = self.w * self.velocity
            cognitive = self.c1 * r1 * (self.p_best_pos - self.pos)
            social = self.c2 * r2 * (self.g_best_pos - self.pos)

            # Update velocity and position
            self.velocity = inertia + cognitive + social
            self.pos += self.velocity
            
            # Keep swarm within bounds
            self.pos = np.clip(self.pos, self.ranges[0], self.ranges[1])

            # Synchronous batch fitness evaluation (Massive NFE speedup)
            fitness = np.apply_along_axis(self.function, 1, self.pos)

            # Identify particles that improved their personal best
            improved_idx = fitness < self.p_best
            self.p_best[improved_idx] = fitness[improved_idx]
            self.p_best_pos[improved_idx] = self.pos[improved_idx]

            # Check for new global best
            min_cur_fitness_idx = np.argmin(self.p_best)
            min_cur_fitness = self.p_best[min_cur_fitness_idx]
            
            if min_cur_fitness < self.g_best:
                self.g_best = min_cur_fitness
                self.g_best_pos = self.p_best_pos[min_cur_fitness_idx].copy()
                
            self.iter += 1
            self.history.append(self.pos.copy().tolist())