import random
from .genetic_algorithm import GA



class GA_Knapsack(GA):
    def __init__(self, weights, values, capacity, **kwargs):
        super().__init__(**kwargs) 
        self.weights = weights
        self.values = values
        self.capacity = capacity
        self.n_items = len(weights)
        
        self.item_ratios = []
        for i in range(self.n_items):
            r = values[i] / weights[i] if weights[i] > 0 else 0
            self.item_ratios.append((i, r))
        
        self.sorted_ratios_asc = sorted(self.item_ratios, key=lambda x: x[1])
        self.sorted_ratios_desc = sorted(self.item_ratios, key=lambda x: x[1], reverse=True)



    def create_individual(self):
        ind = [random.randint(0, 1) for _ in range(self.n_items)]
        return self.repair(ind)



    def repair(self, individual):
        current_weight = sum(individual[i] * self.weights[i] for i in range(self.n_items))
        
        if current_weight > self.capacity:
            for idx, _ in self.sorted_ratios_asc:
                if current_weight <= self.capacity: break
                if individual[idx] == 1:
                    individual[idx] = 0
                    current_weight -= self.weights[idx]
        
        else:
            for idx, _ in self.sorted_ratios_desc:
                if individual[idx] == 0:
                    if current_weight + self.weights[idx] <= self.capacity:
                        individual[idx] = 1
                        current_weight += self.weights[idx]
        return individual



    def calculate_fitness(self, individual):
        total_val = sum(individual[i] * self.values[i] for i in range(self.n_items))
        total_weight = sum(individual[i] * self.weights[i] for i in range(self.n_items))
        return total_val if total_weight <= self.capacity else 0



    def crossover(self, p1, p2):
        if random.random() > 0.9:
            return p1[:], p2[:]
            
        point1 = random.randint(0, self.n_items - 2)
        point2 = random.randint(point1 + 1, self.n_items - 1)
        
        c1 = p1[:point1] + p2[point1:point2] + p1[point2:]
        c2 = p2[:point1] + p1[point1:point2] + p2[point2:]
        
        return self.repair(c1), self.repair(c2)



    def mutate(self, individual):
        for i in range(self.n_items):
            if random.random() < self.mutation_rate:
                individual[i] = 1 - individual[i]
        return self.repair(individual)