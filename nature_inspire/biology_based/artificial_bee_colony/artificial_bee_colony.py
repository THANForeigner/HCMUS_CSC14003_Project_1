import random
import numpy as np
import copy


class DummyBee:
    def __init__(self, coords):
        self.coords = coords

class ABC:
    def __init__(self, function, ranges: list, dimension=-1, swarm_size=-1, limit=-1, max_iteration=-1):
        self.function = function
        self.ranges = ranges
        self.dimension = 2 if dimension == -1 else dimension
        self.swarm_size = 100 if swarm_size == -1 else swarm_size
        self.food_size = int(self.swarm_size / 2)
        self.trial_limit = limit
        self.iter = 0
        self.max_iter = max_iteration
        
        self.lower_bound = ranges[0]
        self.upper_bound = ranges[1]
        
        # Array-based representation
        self.food_sources = np.random.uniform(self.lower_bound, self.upper_bound, (self.food_size, self.dimension))
        self.fitness = self.calculate_fitness(self.food_sources)
        self.trials = np.zeros(self.food_size)
        
        best_idx = np.argmax(self.fitness)
        self.best_source = self.food_sources[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]
        self.history = []

        # Maintain dummy object for backward compatibility with external scripts
        self.best_bee = DummyBee(self.best_source)

    def calculate_fitness(self, pop):
        obj_values = np.apply_along_axis(self.function, 1, pop)
        # Assuming MINIMIZATION problem
        fitness = np.where(obj_values >= 0, 1 / (1 + obj_values), 1 + np.abs(obj_values))
        return fitness

    def run(self):
        self.history.append(self.food_sources.copy().tolist())

        while self.iter < self.max_iter:
            # 1. EMPLOYED BEE PHASE
            # Vectorized exploration
            partners = np.zeros_like(self.food_sources)
            for i in range(self.food_size):
                idxs = [idx for idx in range(self.food_size) if idx != i]
                partners[i] = self.food_sources[np.random.choice(idxs)]
            
            j = np.random.randint(0, self.dimension, self.food_size)
            phi = np.random.uniform(-1, 1, self.food_size)
            
            new_sources = self.food_sources.copy()
            new_sources[np.arange(self.food_size), j] = self.food_sources[np.arange(self.food_size), j] + phi * (self.food_sources[np.arange(self.food_size), j] - partners[np.arange(self.food_size), j])
            new_sources = np.clip(new_sources, self.lower_bound, self.upper_bound)
            
            new_fitness = self.calculate_fitness(new_sources)
            
            improved = new_fitness > self.fitness
            self.food_sources[improved] = new_sources[improved]
            self.fitness[improved] = new_fitness[improved]
            self.trials[improved] = 0
            self.trials[~improved] += 1
            
            # 2. ONLOOKER BEE PHASE
            total_fitness = np.sum(self.fitness)
            if total_fitness > 0:
                probs = self.fitness / total_fitness
            else:
                probs = np.ones(self.food_size) / self.food_size
                
            selected_sources_idx = np.random.choice(self.food_size, size=self.food_size, p=probs)
            
            partners_onlooker = np.zeros_like(self.food_sources)
            for i, selected_idx in enumerate(selected_sources_idx):
                idxs = [idx for idx in range(self.food_size) if idx != selected_idx]
                partners_onlooker[i] = self.food_sources[np.random.choice(idxs)]
                
            j_onlooker = np.random.randint(0, self.dimension, self.food_size)
            phi_onlooker = np.random.uniform(-1, 1, self.food_size)
            
            new_onlooker_sources = self.food_sources[selected_sources_idx].copy()
            new_onlooker_sources[np.arange(self.food_size), j_onlooker] = self.food_sources[selected_sources_idx, j_onlooker] + phi_onlooker * (self.food_sources[selected_sources_idx, j_onlooker] - partners_onlooker[np.arange(self.food_size), j_onlooker])
            new_onlooker_sources = np.clip(new_onlooker_sources, self.lower_bound, self.upper_bound)
            
            new_onlooker_fitness = self.calculate_fitness(new_onlooker_sources)
            
            for i, selected_idx in enumerate(selected_sources_idx):
                if new_onlooker_fitness[i] > self.fitness[selected_idx]:
                    self.food_sources[selected_idx] = new_onlooker_sources[i]
                    self.fitness[selected_idx] = new_onlooker_fitness[i]
                    self.trials[selected_idx] = 0
                else:
                    self.trials[selected_idx] += 1

            # 3. SCOUT BEE PHASE
            abandoned = self.trials > self.trial_limit
            num_abandoned = np.sum(abandoned)
            if num_abandoned > 0:
                self.food_sources[abandoned] = np.random.uniform(self.lower_bound, self.upper_bound, (num_abandoned, self.dimension))
                self.fitness[abandoned] = self.calculate_fitness(self.food_sources[abandoned])
                self.trials[abandoned] = 0

            # Update Best
            best_cur_idx = np.argmax(self.fitness)
            if self.fitness[best_cur_idx] > self.best_fitness:
                self.best_fitness = self.fitness[best_cur_idx]
                self.best_source = self.food_sources[best_cur_idx].copy()
                self.best_bee.coords = self.best_source.copy()

            self.iter += 1
            self.history.append(self.food_sources.copy().tolist())