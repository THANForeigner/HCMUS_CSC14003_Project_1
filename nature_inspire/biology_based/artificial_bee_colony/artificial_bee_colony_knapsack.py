import random
import numpy as np

class Bee:
    def __init__(self, coords, fitness):
                self.coords = coords
                self.fitness = fitness
class ABC_Knapsack:
    def __init__(self, items: int, capacity: int, weight: list, cost: list, swarm_size=-1, limit=-1, max_iteration=-1):
        self.HARD_LIMIT_ITEMS = 10000
        if items > self.HARD_LIMIT_ITEMS:
            raise ValueError(f"Knapsack too large for ABC (N={items} > {self.HARD_LIMIT_ITEMS}).")

        self.items = items
        self.capacity = capacity
        self.weight = np.array(weight)
        self.cost = np.array(cost)
        
        self.swarm_size = 100 if swarm_size == -1 else swarm_size
        self.food_size = int(self.swarm_size / 2)
        self.trial_limit = 3 if limit == -1 else limit
        self.max_iter = max_iteration
        
        # Khởi tạo Food Sources (Binary Matrix)
        self.food_sources = np.random.randint(2, size=(self.food_size, self.items))
        self.fitness = self.calculate_fitness(self.food_sources)
        self.trials = np.zeros(self.food_size)
        
        best_idx = np.argmax(self.fitness)
        self.best_fitness = self.fitness[best_idx]
        self.best_source = self.food_sources[best_idx].copy()
        self.best_bee = Bee(self.best_source, self.best_fitness)

    def calculate_fitness(self, pop):
        if pop.ndim == 1:
            total_weight = np.sum(pop * self.weight)
            total_cost = np.sum(pop * self.cost)
            return total_cost if total_weight <= self.capacity else 0
        
        total_weights = np.dot(pop, self.weight)
        total_costs = np.dot(pop, self.cost)
        
        # Nếu vượt quá capacity, fitness = 0
        fitness = np.where(total_weights <= self.capacity, total_costs, 0)
        return fitness

    def run(self):
        for _ in range(self.max_iter):
            # 1. Pha ong thợ
            partners_idx = np.array([random.choice([idx for idx in range(self.food_size) if idx != i]) for i in range(self.food_size)])
            partners = self.food_sources[partners_idx]
            
            j = np.random.randint(0, self.items, self.food_size)
            phi = np.random.uniform(-1, 1, self.food_size)
            
            curr_vals = self.food_sources[np.arange(self.food_size), j]
            part_vals = partners[np.arange(self.food_size), j]
            velocities = curr_vals + phi * (curr_vals - part_vals)
            probs = 1.0 / (1.0 + np.exp(-velocities))
            
            new_sources = self.food_sources.copy()
            new_bits = (np.random.random(self.food_size) < probs).astype(int)
            new_sources[np.arange(self.food_size), j] = new_bits
            
            new_fitness = self.calculate_fitness(new_sources)
            
            # Chọn lựa theo tham lam
            improved = new_fitness > self.fitness
            self.food_sources[improved] = new_sources[improved]
            self.fitness[improved] = new_fitness[improved]
            self.trials[improved] = 0
            self.trials[~improved] += 1

            # 2. Pha ong quan sát
            sum_fit = np.sum(self.fitness)
            if sum_fit == 0:
                prob_onlooker = np.ones(self.food_size) / self.food_size
            else:
                prob_onlooker = self.fitness / sum_fit
            
            # Chọn nguồn thức ăn
            selected_indices = np.random.choice(self.food_size, size=self.food_size, p=prob_onlooker)
            selected_sources = self.food_sources[selected_indices]
            
            # Chọn ong hỗ trợ
            onlooker_partners_idx = np.array([random.choice([idx for idx in range(self.food_size) if idx != s_idx]) for s_idx in selected_indices])
            onlooker_partners = self.food_sources[onlooker_partners_idx]
            
            j_on = np.random.randint(0, self.items, self.food_size)
            phi_on = np.random.uniform(-1, 1, self.food_size)
            
            on_curr_vals = selected_sources[np.arange(self.food_size), j_on]
            on_part_vals = onlooker_partners[np.arange(self.food_size), j_on]
            on_velocities = on_curr_vals + phi_on * (on_curr_vals - on_part_vals)
            on_probs = 1.0 / (1.0 + np.exp(-on_velocities))
            
            new_onlooker_sources = selected_sources.copy()
            new_on_bits = (np.random.random(self.food_size) < on_probs).astype(int)
            new_onlooker_sources[np.arange(self.food_size), j_on] = new_on_bits
            
            new_on_fitness = self.calculate_fitness(new_onlooker_sources)
            
            # Cập nhật nguồn thức ăn
            for i, s_idx in enumerate(selected_indices):
                if new_on_fitness[i] > self.fitness[s_idx]:
                    self.food_sources[s_idx] = new_onlooker_sources[i]
                    self.fitness[s_idx] = new_on_fitness[i]
                    self.trials[s_idx] = 0
                else:
                    self.trials[s_idx] += 1

            # 3. Pha ong trinh sát
            abandoned = self.trials > self.trial_limit
            num_abandoned = np.sum(abandoned)
            if num_abandoned > 0:
                self.food_sources[abandoned] = np.random.randint(2, size=(num_abandoned, self.items))
                self.fitness[abandoned] = self.calculate_fitness(self.food_sources[abandoned])
                self.trials[abandoned] = 0

            best_idx = np.argmax(self.fitness)
            if self.fitness[best_idx] > self.best_fitness:
                self.best_fitness = self.fitness[best_idx]
                self.best_source = self.food_sources[best_idx].copy()
                self.best_bee.coords = self.best_source
                self.best_bee.fitness = self.best_fitness