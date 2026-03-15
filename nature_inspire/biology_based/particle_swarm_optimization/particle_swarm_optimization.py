import numpy as np
import random


class PSO:
    def __init__(self, function, dimension=-1, ranges=None, swarm_size=-1,
                 w=-1, c1=-1, c2=-1, max_interation=-1):
        self.function = function
        self.dim = 2 if dimension == -1 else dimension
        self.ranges = ranges
        self.swarm_size = 50 if swarm_size == -1 else swarm_size

        # Hằng số quan trọng cho PSO
        self.w = 0.729 if w == -1 else w  # Quán tính
        self.c1 = 1.49445 if c1 == -1 else c1  # Hằng số nhận thức
        self.c2 = 1.49445 if c2 == -1 else c2  # Hằng số xã hội

        self.max_iter = 1000 if max_interation == -1 else max_interation
        self.iter = 0
        self.history = []

        self.pos = np.random.uniform(self.ranges[0], self.ranges[1], (self.swarm_size, self.dim))
        self.velocity = np.random.uniform(-1, 1, (self.swarm_size, self.dim))      
        self.p_best_pos = self.pos.copy()
        self.p_best = np.apply_along_axis(self.function, 1, self.pos)
        
        best_idx = np.argmin(self.p_best)
        self.g_best = self.p_best[best_idx]
        self.g_best_pos = self.p_best_pos[best_idx].copy()

    def run(self):
        self.history.append(self.pos.copy().tolist())

        while self.iter < self.max_iter:
            # Sinh random cả đàn 
            r1 = np.random.rand(self.swarm_size, self.dim)
            r2 = np.random.rand(self.swarm_size, self.dim)

            # Tính toán và cập nhật vị trí của hạt cá thể
            inertia = self.w * self.velocity
            cognitive = self.c1 * r1 * (self.p_best_pos - self.pos)
            social = self.c2 * r2 * (self.g_best_pos - self.pos)
            self.velocity = inertia + cognitive + social
            self.pos += self.velocity
            
            self.pos = np.clip(self.pos, self.ranges[0], self.ranges[1])
            fitness = np.apply_along_axis(self.function, 1, self.pos)

            improved_idx = fitness < self.p_best
            self.p_best[improved_idx] = fitness[improved_idx]
            self.p_best_pos[improved_idx] = self.pos[improved_idx]

            min_cur_fitness_idx = np.argmin(self.p_best)
            min_cur_fitness = self.p_best[min_cur_fitness_idx]
            
            if min_cur_fitness < self.g_best:
                self.g_best = min_cur_fitness
                self.g_best_pos = self.p_best_pos[min_cur_fitness_idx].copy()
                
            self.iter += 1
            self.history.append(self.pos.copy().tolist())