import math
import random


class SimulatedAnnealingContinuous:
    def __init__(
        self,
        ranges,
        function,
        T=-1,
        dim=-1,
        step_size=-1,
        alpha=-1,
        stopping_T=-1,
        stopping_iter=-1,
    ):
        self.ranges = ranges
        self.function = function
        self.step_size = 0.1 if step_size == -1 else step_size
        self.D = 2 if dim == -1 else dim
        self.T = 1000 if T == -1 else T  # Temperature
        self.save_T = self.T  # Store temparature for batch simulated annealling
        self.alpha = alpha if alpha >= 0 and alpha <= 1 else 0.995
        self.stopping_T = 1e-8 if stopping_T == -1 else stopping_T
        self.stopping_iter = 100000 if stopping_iter == -1 else stopping_iter
        self.iteration = 1
        self.cur_coords = []
        self.cur_result = float("Inf")
        self.best_result = float("Inf")
        self.best_coords = []
        self.energy_list = []
        self.coords_history = []

    def get_neighbour(self):
        new_coords = []
        for coord in self.cur_coords:
            noise = random.uniform(-1, 1) * self.step_size
            next = coord + noise
            next = max(self.ranges[0], min(self.ranges[1], next))
            new_coords.append(next)
        return new_coords

    def simulated_annealling(self):
        self.cur_coords = []
        self.iteration = 1
        self.T = self.save_T
        for i in range(0, self.D):
            self.cur_coords.append(random.uniform(self.ranges[0], self.ranges[1]))
        self.cur_result = self.function(self.cur_coords)
        self.energy_list.append(self.cur_result)
        self.coords_history.append(self.cur_coords)
        self.best_result = self.cur_result
        self.best_coords = list(self.cur_coords)

        while self.T >= self.stopping_T and self.iteration < self.stopping_iter:
            new_coords = self.get_neighbour()
            new_result = self.function(new_coords)
            delta_energy = new_result - self.cur_result
            if new_result < self.cur_result:
                self.cur_coords = new_coords
                self.cur_result = new_result
                if new_result < self.best_result:
                    self.best_result = new_result
                    self.best_coords = new_coords
            else:
                p_accept = math.exp(-delta_energy / self.T)
                if random.uniform(0,1) < p_accept:
                    self.cur_coords = new_coords
                    self.cur_result = new_result
            self.T *= self.alpha
            self.iteration += 1
            self.energy_list.append(self.cur_result)
            self.coords_history.append(self.cur_coords)

    def batch_annealing(self, times=10):
        global_best_result = float("inf")
        global_best_coords = None

        for _ in range(times):
            self.simulated_annealling()

            if self.best_result < global_best_result:
                global_best_result = self.best_result
                global_best_coords = list(self.best_coords)

        self.best_result = global_best_result
        self.best_coords = global_best_coords

