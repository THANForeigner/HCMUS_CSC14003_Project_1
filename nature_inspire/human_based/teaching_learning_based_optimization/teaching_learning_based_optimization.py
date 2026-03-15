import numpy as np
from problems import continuous_functions

THRESHOLD_FITNESS = 0.0
TF = 2


class TLBO:
    def __init__(
            self, lower_bound, upper_bound, num_parameters, population_size
    ) -> None:
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.num_parameters = num_parameters
        self.population_size = population_size
        self.population = np.random.uniform(lower_bound, upper_bound, (population_size, num_parameters))
        self.optimized_function = None

    def set_optimization_function(self, f):
        self.optimized_function = f
        self.fitness_values = np.apply_along_axis(self.optimized_function, 1, self.population)
        self.history = []

    # Chọn ra cá thể (ứng viên) tốt nhất để làm giáo viên,
    # Sau đó, mỗi cá thể sẽ học hỏi từ giáo viên này.
    # Giá trị của cá thể sẽ dịch chuyển hướng về phía giá trị của giáo viên
    def teaching_phase(self):
        idx = np.argmin(self.fitness_values)
        teacher = self.population[idx]
        mean_value = np.mean(self.population, axis=0)

        delta = np.random.uniform(0, 1, (self.population_size, self.num_parameters))
        new_population = self.population + delta * (teacher - TF * mean_value)
        new_population = np.clip(new_population, self.lower_bound, self.upper_bound)

        new_fitnesses = np.apply_along_axis(self.optimized_function, 1, new_population)
        
        better_idx = new_fitnesses < self.fitness_values
        self.population[better_idx] = new_population[better_idx]
        self.fitness_values[better_idx] = new_fitnesses[better_idx]

    def learning_phase(self):
        # Chọn bạn học chung random
        partner_indices = np.zeros(self.population_size, dtype=int)
        for i in range(self.population_size):
            idxs = [idx for idx in range(self.population_size) if idx != i]
            partner_indices[i] = np.random.choice(idxs)
            
        partners = self.population[partner_indices]
        partner_fitnesses = self.fitness_values[partner_indices]
        
        delta = np.random.uniform(0, 1, (self.population_size, self.num_parameters))
        
        # Khi người học tốt hơn người bạn học chung
        better_mask = (self.fitness_values < partner_fitnesses)[:, np.newaxis]
        
        new_population = np.where(better_mask, 
                                  self.population + delta * (self.population - partners),
                                  self.population + delta * (partners - self.population))
                                  
        new_population = np.clip(new_population, self.lower_bound, self.upper_bound)
        new_fitnesses = np.apply_along_axis(self.optimized_function, 1, new_population)
        
        better_idx = new_fitnesses < self.fitness_values
        self.population[better_idx] = new_population[better_idx]
        self.fitness_values[better_idx] = new_fitnesses[better_idx]

    def run(self, num_iteration):
        for _ in range(num_iteration):
            self.history.append([list(p) for p in self.population])
            best_fitness = min(self.fitness_values)
            self.teaching_phase()
            self.learning_phase()
            new_best_fitness = min(self.fitness_values)
            if abs(new_best_fitness - best_fitness) < THRESHOLD_FITNESS:
                break

        self.history.append([list(p) for p in self.population])

        best_idx = np.argmin(self.fitness_values)
        return self.fitness_values[best_idx], self.population[best_idx]