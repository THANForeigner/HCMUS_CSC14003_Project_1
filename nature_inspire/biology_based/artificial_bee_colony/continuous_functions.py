import random
import numpy as np
import copy

class Bee:
    def __init__(self, bounds, function, limit = -1, dimension = -1, coords = None):
        self.bounds = bounds
        self.function = function
        self.dimension = 2 if dimension == -1 else dimension
        self.coords = self.randposition() if coords is None else coords
        self.fitness = self.getfitness(self.coords)
        self.trial = 0
        self.limit = 3 if limit == -1 else limit
    
    def randposition(self):
        return np.array([
            self.bounds[0] + random.uniform(0, 1) * (self.bounds[1] - self.bounds[0])
            for _ in range(self.dimension)
        ])
    
    def getfitness(self, coords):
        fi = self.function(coords.tolist())
        fitness = None
        if fi >= 0:
            fitness = 1 / ( 1 + fi )
        else:
            fitness = 1 + abs(fi)
        return fitness
    
    def _explore(self, partner_coords):
        des_coords = partner_coords
        new_coords = self.coords + random.uniform(-1,1) * (self.coords - des_coords)
        new_coords = np.clip(new_coords, self.bounds[0], self.bounds[1])
        new_fitness = self.getfitness(new_coords)
        if self.fitness < new_fitness:
            self.fitness = new_fitness
            self.coords = new_coords
            self.trial = 0
            return True
        else:
            self.trial += 1 
            return False
    
    def reset(self):
        self.coords = self.randposition()
        self.fitness = self.getfitness(self.coords)
        self.trial = 0
        
class EmployeeBee (Bee):
    def explore(self, partner_bee):
        self._explore(partner_bee.coords)
    
class OnlookerBee (Bee):
    def explore(self, selected_bee, partner_bee):
        self.coords = selected_bee.coords
        self.fitness = selected_bee.fitness
        self.trial = selected_bee.trial 
        is_improved = self._explore(partner_bee.coords)
        if is_improved:
            selected_bee.coords = self.coords
            selected_bee.fitness = self.fitness
            selected_bee.trial = 0
        else:
            selected_bee.trial += 1
        
class ArtificialBeeColony:
    def __init__ (self, function, ranges: list, dimension = -1, swarm_size = -1, limit = -1, max_iteration = -1):
        self.function = function
        self.ranges = ranges
        self.dimension = 2 if dimension == -1 else dimension
        self.swarm_size = 100 if swarm_size == -1 else swarm_size
        self.food_size = int(self.swarm_size / 2)
        self.trial_limit = limit
        self.iter = 0
        self.max_iter = max_iteration
        self.best_bee = Bee(self.ranges, self.function, self.trial_limit, self.dimension)
        self.employee_bees = [
            EmployeeBee(self.ranges, self.function, self.trial_limit, self.dimension)
            for _ in range (0, self.food_size)
        ]
        self.onlooker_bees = [
            OnlookerBee(self.ranges, self.function, self.trial_limit, self.dimension)
            for _ in range(self.food_size)
        ]
        self.fitness_history = []
        
    def artificial_bee_colony(self):
        while self.iter < self.max_iter:
            for i,bee in enumerate(self.employee_bees):
                candidates = [b for idx, b in enumerate(self.employee_bees) if idx != i]
                partner = random.choice(candidates)
                bee.explore(partner)
        
            overall_fitness = sum(bee.fitness for bee in self.employee_bees)
            
            probs = [bee.fitness/overall_fitness for bee in self.employee_bees]
            selected_sources = random.choices(range(self.food_size), weights=probs, k=self.food_size)
            
            for onlooker, idx in zip(self.onlooker_bees, selected_sources):
                selected_bee = self.employee_bees[idx]
                candidates = [b for i, b in enumerate(self.employee_bees) if i != idx]
                partner_bee = random.choice(candidates)
                onlooker.explore(selected_bee, partner_bee)
            
            for bee in self.employee_bees:
                if bee.trial > self.trial_limit:
                    bee.reset()
                    
            current_best_bee = max(self.employee_bees, key=lambda bee:bee.fitness)
            if current_best_bee.fitness > self.best_bee.fitness:
                self.best_bee = copy.deepcopy(current_best_bee)    
            
            self.iter += 1