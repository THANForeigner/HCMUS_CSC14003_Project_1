import numpy as np
from nature_inspire import continuous_functions

THRESHOLD_FITNESS = 0.01
TF = 2


class TeachingLearingBasedOptimization:
    def __init__(
        self, lower_bound, upper_bound, num_parameters, population_size
    ) -> None:
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.num_parameters = num_parameters
        self.population_size = population_size
        self.population = []
        for _ in range(population_size):
            # spawn a member with randomizely with number of num_parameters
            member = np.random.uniform(lower_bound, upper_bound, num_parameters)
            self.population.append(member)
        self.optimized_function = None

    def set_optimization_function(self, f):
        self.optimized_function = f
        self.fitness_values = [self.optimized_function(it) for it in self.population]

    # Select the best candidate to be a teacher,
    # Then each candidate will learn from a teacher.
    # candidate value will move forward to a teacher's value
    def teaching_phase(self):
        idx = np.argmin(self.fitness_values)
        teacher = self.population[idx]
        mean_value = np.mean(self.population, axis=0)

        for i, candidate in enumerate(self.population):
            delta = np.random.uniform(0, 1, self.num_parameters)
            new_candidate = candidate + delta * (teacher - TF * mean_value)
            new_candidate = np.clip(new_candidate, self.lower_bound, self.upper_bound)

            new_fitness = self.optimized_function(new_candidate)
            if new_fitness < self.fitness_values[i]:
                self.population[i] = new_candidate
                self.fitness_values[i] = new_fitness

    # Pick a partner for each candidate
    # Then each candidate will learn from their partner
    # candidate value will move forward to their partner if fitness_value if better else do opposite
    def learning_phase(self):
        for i in range(self.population_size):
            partner_index = np.random.choice(
                [idx for idx in range(self.population_size) if idx != i]
            )
            candidate = self.population[i]
            partner = self.population[partner_index]
            delta = np.random.uniform(0, 1, self.num_parameters)

            if self.fitness_values[i] < self.fitness_values[partner_index]:
                new_candidate = candidate + delta * (candidate - partner)
            else:
                new_candidate = candidate + delta * (partner - candidate)

            new_candidate = np.clip(new_candidate, self.lower_bound, self.upper_bound)
            new_fitness = self.optimized_function(new_candidate)
            if new_fitness < self.fitness_values[i]:
                self.population[i] = new_candidate
                self.fitness_values[i] = new_fitness

    def solve(self, num_iteration):
        for _ in range(num_iteration):
            best_fitness = min(self.fitness_values)
            self.teaching_phase()
            self.learning_phase()
            new_best_fitness = min(self.fitness_values)
            if abs(new_best_fitness - best_fitness) < THRESHOLD_FITNESS:
                break

        best_idx = np.argmin(self.fitness_values)
        return self.fitness_values[best_idx], self.population[best_idx]
