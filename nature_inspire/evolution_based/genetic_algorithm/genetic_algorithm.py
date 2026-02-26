from abc import ABC, abstractmethod
import random



class GA(ABC):
    def __init__(self, pop_size=50, generations=100, mutation_rate=0.1, elitism=True):
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.population = []

    @abstractmethod
    def create_individual(self):
        pass

    @abstractmethod
    def calculate_fitness(self, individual):
        pass

    @abstractmethod
    def crossover(self, parent1, parent2):
        pass

    @abstractmethod
    def mutate(self, individual):
        pass

    def selection(self):
        k = 3
        competitors = random.sample(self.population, k)
        competitors.sort(key=lambda x: x[1], reverse=True)
        return competitors[0][0]

    def run(self):
        self.population = [(self.create_individual(), 0) for _ in range(self.pop_size)]
        self.population = [(ind, self.calculate_fitness(ind)) for ind, _ in self.population]
        
        history = []

        for gen in range(self.generations):
            self.population.sort(key=lambda x: x[1], reverse=True)
            
            best_fitness = self.population[0][1]
            history.append(best_fitness)
            
            new_population = []
            if self.elitism:
                new_population.append(self.population[0])
            
            while len(new_population) < self.pop_size:
                p1 = self.selection(); p2 = self.selection()
                c1, c2 = self.crossover(p1, p2)
                new_population.append((self.mutate(c1), self.calculate_fitness(c1)))
                if len(new_population) < self.pop_size:
                    new_population.append((self.mutate(c2), self.calculate_fitness(c2)))
            
            self.population = new_population
            
        return self.population[0], history