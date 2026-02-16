import numpy as np

THRESHOLD_FITNESS = 0.01


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

    def set_optimization_function(self, f):
        self.optimized_function = f
        self.current_fitness = self.optimized_function(self.current_state)

    def climbing_phase(self):
        # Try to find a better neighbor
        # We sample a few neighbors to find a direction of improvement
        num_neighbors = 20
        
        for _ in range(num_neighbors):
            delta = np.random.uniform(-1, 1, self.num_parameters) * self.step_size
            neighbor = self.current_state + delta
            neighbor = np.clip(neighbor, self.lower_bound, self.upper_bound)
            
            new_fitness = self.optimized_function(neighbor)
            if new_fitness < self.current_fitness:
                self.current_state = neighbor
                self.current_fitness = new_fitness

    def solve(self, num_iteration):
        for _ in range(num_iteration):
            best_fitness = self.current_fitness
            self.climbing_phase()
            new_best_fitness = self.current_fitness
            if abs(new_best_fitness - best_fitness) < THRESHOLD_FITNESS:
                break

        return self.current_fitness, self.current_state
