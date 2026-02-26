import random
import math
import numpy as np
import copy

class Bee:
    def __init__(self, items: int, weight: list, cost: list, capacity: int, limit = -1, coords = None):
        self.items = items
        self.weight = np.array(weight)
        self.cost = np.array(cost)
        self.capacity = capacity
        self.coords = self.randposition() if coords is None else coords
        self.fitness = self.getfitness(self.coords)
        self.trial = 0
        self.limit = 3 if limit == -1 else limit
    
    def randposition(self):
        return np.random.randint(2, size=self.items)
    
    def getfitness(self, coords):
        #if the weight is higher than the capacity, give up that solution by give fitness 0
        total_weight = np.sum(coords * self.weight)
        total_cost = np.sum(coords * self.cost)
        if total_weight > self.capacity:
            return 0
        return total_cost
    
    def _explore(self, partner_coords):
        
        #Change one dimension to make a variation
        #Then apply the search equation for the bees
        new_coords = self.coords.copy()
        j = random.randint(0, self.items - 1)
        phi = random.uniform(-1, 1)
        velocity = self.coords[j] + phi * (self.coords[j] - partner_coords[j])
        prob = 1 / (1 + math.exp(-velocity))
        if random.random() < prob:
            new_coords[j] = 1
        else:
            new_coords[j] = 0
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
        #Onlooker bees observe employed bees to make decision for the food sources
        self.coords = selected_bee.coords.copy()
        self.fitness = selected_bee.fitness
        self.trial = selected_bee.trial 
        is_improved = self._explore(partner_bee.coords)
        if is_improved:
            selected_bee.coords = self.coords.copy()
            selected_bee.fitness = self.fitness
        selected_bee.trial = self.trial
        
class ABC_Knapsack:
    def __init__ (self, items: int, capacity: int, weight: list, cost: list, swarm_size = -1, limit = -1, max_iteration = -1):
        self.items = items
        self.capacity = capacity
        self.weight = weight
        self.cost = cost
        self.decision = [0 for i in range (0, self.items)]
        self.swarm_size = 100 if swarm_size == -1 else swarm_size
        self.food_size = int(self.swarm_size / 2)
        self.trial_limit = limit
        self.iter = 0
        self.max_iter = max_iteration
        
        # Initiate food sources and employed bees at the food sources
        self.employee_bees = [
            EmployeeBee(self.items, self.weight, self.cost, self.capacity, self.trial_limit)
            for _ in range (0, self.food_size)
        ]
        
        # Initiate onlooker bees
        self.onlooker_bees = [
            OnlookerBee(self.items, self.weight, self.cost, self.capacity, self.trial_limit)
            for _ in range(self.food_size)
        ]
        
        # Find the best food source in the initial population
        initial_best = max(self.employee_bees, key=lambda bee: bee.fitness)
        self.best_bee = copy.deepcopy(initial_best)
        self.fitness_history = []
        
    def run(self):
        while self.iter < self.max_iter:
            
            # 1. EMPLOYED BEE PHASE
            # Every employed bee explores near its current food source
            for i,bee in enumerate(self.employee_bees):
                candidates = [b for idx, b in enumerate(self.employee_bees) if idx != i]
                partner = random.choice(candidates)
                bee.explore(partner)

            # Calculate probabilities for Roulette Wheel Selection
            overall_fitness = sum(bee.fitness for bee in self.employee_bees)
            if overall_fitness == 0:
                probs = [1.0 / self.food_size] * self.food_size
            else:
                probs = [bee.fitness/overall_fitness for bee in self.employee_bees]
            
            # Select food sources based on their fitness weights
            selected_sources = random.choices(range(self.food_size), weights=probs, k=self.food_size)
            
            # 2. ONLOOKER BEE PHASE
            # Onlookers are deployed to the selected sources
            for onlooker, idx in zip(self.onlooker_bees, selected_sources):
                selected_bee = self.employee_bees[idx]
                candidates = [b for i, b in enumerate(self.employee_bees) if i != idx]
                partner_bee = random.choice(candidates)
                onlooker.explore(selected_bee, partner_bee)
            
            # 3. SCOUT BEE PHASE
            # Abandon food sources that have exceeded the trial limit
            for bee in self.employee_bees:
                if bee.trial > self.trial_limit:
                    bee.reset()
                    
            current_best_bee = max(self.employee_bees, key=lambda bee:bee.fitness)
            if current_best_bee.fitness > self.best_bee.fitness:
                self.best_bee = copy.deepcopy(current_best_bee)    
            
            self.iter += 1