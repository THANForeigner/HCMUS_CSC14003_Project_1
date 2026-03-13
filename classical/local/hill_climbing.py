import numpy as np

THRESHOLD_FITNESS = 1e-9


class HillClimbing:
    def __init__(
            self, lower_bound, upper_bound, num_parameters, step_size=0.1
    ) -> None:
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.num_parameters = num_parameters
        self.step_size = step_size
        self.current_state = np.random.uniform(lower_bound, upper_bound, num_parameters)
        self.optimized_function = None
        self.current_fitness = None
        self.history = []

    def set_optimization_function(self, f):
        self.optimized_function = f
        self.current_fitness = self.optimized_function(self.current_state)

    def climbing_phase(self):
        # Try to find a better neighbor
        # We sample a few neighbors to find a direction of improvement
        num_neighbors = 20

        deltas = np.random.uniform(-1, 1, (num_neighbors, self.num_parameters)) * self.step_size
        neighbors = self.current_state + deltas
        neighbors = np.clip(neighbors, self.lower_bound, self.upper_bound)
        
        fitnesses = np.apply_along_axis(self.optimized_function, 1, neighbors)
        best_idx = np.argmin(fitnesses)
        best_new_fitness = fitnesses[best_idx]
        
        if best_new_fitness < self.current_fitness:
            self.current_state = neighbors[best_idx]
            self.current_fitness = best_new_fitness

    def run(self, num_iteration):
        for _ in range(num_iteration):
            self.history.append(self.current_state.copy())

            best_fitness = self.current_fitness
            self.climbing_phase()
            new_best_fitness = self.current_fitness
            if abs(new_best_fitness - best_fitness) < THRESHOLD_FITNESS:
                break

        self.history.append(self.current_state.copy())

        return self.current_fitness, self.current_state