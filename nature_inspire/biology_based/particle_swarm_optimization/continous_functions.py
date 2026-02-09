import numpy as np
import random
class Particle:
    def __init__(self, dimensions: int, bounds: list, function):
        self.dim = dimensions
        self.bounds = bounds
        self.pos = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)
        self.velocity = np.random.uniform(-1, 1, self.dim)
        self.p_best = float("Inf")
        self.p_best_pos = None
        self.f = function
        self.cur_fitness = self.get_fitness()
    
    def move(self):
        self.pos += self.velocity
        self.pos = np.clip(self.pos, self.bounds[0], self.bounds[1])
        self.get_fitness()
    
    def next_velocity(self, velocity: list):
        self.velocity = np.array(velocity)
    
    def get_fitness(self):
        fitness = self.f(self.pos.tolist())
        if fitness < self.p_best:
            self.p_best = fitness
            self.p_best_pos = self.pos.copy()
        return fitness

class ParticleSwarmOptimization:
    def __init__(self, function, dimension = -1, ranges = None, swarm_size = -1,
                 w = -1, c1 = -1, c2 = -1, max_interation = -1):
        self.function = function
        self.dim = 2 if dimension == -1 else dimension
        self.ranges = ranges
        self.swarm_size = 50 if swarm_size == -1 else swarm_size
        self.w = 0.729 if w == -1 else w
        self.c1 = 1.49445 if c1 == -1 else c1
        self.c2 = 1.49445 if c2 == -1 else c2
        self.max_iter = 1000 if max_interation == -1 else max_interation
        self.iter = 0
        self.swarm = []
        self.g_best = float("Inf")
        self.g_best_pos = []
        for i in range (0, self.swarm_size):
            particle = Particle(self.dim, self.ranges, self.function)
            if particle.p_best < self.g_best:
                self.g_best = particle.p_best
                self.g_best_pos = particle.p_best_pos.tolist().copy()
            self.swarm.append(particle)
            
    def particle_swarm_optimization(self):
        while self.iter < self.max_iter:
            #compute new velocity
            for part in self.swarm:
                next_vel = []
                cur_vel = part.velocity.tolist().copy()
                cur_p_best_pos = part.p_best_pos.tolist().copy()
                cur_pos = part.pos.tolist().copy()
                for k in range (self.dim):
                    r1 = random.random()
                    r2 = random.random()
                    inertia = self.w * cur_vel[k]
                    cognitive = self.c1 * r1 * (cur_p_best_pos[k] - cur_pos[k])
                    social = self.c2 * r2 * (self.g_best_pos[k] - cur_pos[k])
                    vel_coord = (inertia + cognitive + social)
                    next_vel.append(vel_coord)
                part.next_velocity(next_vel)
                part.move()
                if part.p_best < self.g_best:
                    self.g_best = part.p_best
                    self.g_best_pos = part.p_best_pos.tolist().copy()
            self.iter += 1
