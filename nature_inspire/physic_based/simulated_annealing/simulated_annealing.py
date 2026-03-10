import math
import random


class SA:
    def __init__(
            self,
            bounds,
            function,
            T=-1,
            dim=-1,
            step_size=-1,
            alpha=-1,
            stopping_T=-1,
            stopping_iter=-1,
    ):
        self.bounds = bounds
        self.function = function
        self.step_size = 0.1 if step_size == -1 else step_size
        self.D = 2 if dim == -1 else dim  # Dimension
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
        self.history = []

    # Get the next neighbour state by move the current state forward or behind
    def get_neighbour(self):
        new_coords = []
        for coord in self.cur_coords:
            noise = random.uniform(-1, 1) * self.step_size
            next = coord + noise
            next = max(self.bounds[0], min(self.bounds[1], next))
            new_coords.append(next)
        return new_coords

    def simulated_annealing(self):
        self.cur_coords = []
        self.iteration = 1
        self.T = self.save_T

        # Generate the initial state
        for i in range(0, self.D):
            self.cur_coords.append(random.uniform(self.bounds[0], self.bounds[1]))
        self.cur_result = self.function(self.cur_coords)

        self.energy_list.append(self.cur_result)
        self.history.append(list(self.cur_coords))
        self.best_result = self.cur_result
        self.best_coords = list(self.cur_coords)

        while self.T >= self.stopping_T and self.iteration < self.stopping_iter:
            new_coords = self.get_neighbour()
            new_result = self.function(new_coords)
            delta_energy = new_result - self.cur_result

            # Delta energy < 0. Therefore, always accept the more optimal result
            if new_result < self.cur_result:
                self.cur_coords = new_coords
                self.cur_result = new_result
                if new_result < self.best_result:
                    self.best_result = new_result
                    self.best_coords = new_coords

            # Delta energy >= 0. Consider accepting the sub-optimal result using the Metropolis acceptance criterion
            else:
                p_accept = math.exp(-delta_energy / self.T)
                if random.uniform(0, 1) < p_accept:
                    self.cur_coords = new_coords
                    self.cur_result = new_result

            # Perform the cooling schedule
            self.T *= self.alpha

            self.iteration += 1
            self.energy_list.append(self.cur_result)
            self.history.append(list(self.cur_coords))

    # Batch annealing to repeat simulated annealing in order to increase the accuracy of the result
    def batch_annealing(self, times=10):
        global_best_result = float("inf")
        global_best_coords = None

        for _ in range(times):
            self.simulated_annealing()

            if self.best_result < global_best_result:
                global_best_result = self.best_result
                global_best_coords = list(self.best_coords)

        self.best_result = global_best_result
        self.best_coords = global_best_coords

    def run(self, times=10):
        self.batch_annealing(times)