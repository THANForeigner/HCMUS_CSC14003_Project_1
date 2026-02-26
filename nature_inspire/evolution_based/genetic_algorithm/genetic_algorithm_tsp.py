import random
from genetic_algorithm import GA



class GA_TSP(GA):
    def __init__(self, distance_matrix, local_search_prob=0.5, **kwargs):
        super().__init__(**kwargs)
        self.matrix = distance_matrix
        self.n_cities = len(distance_matrix)
        self.local_search_prob = local_search_prob



    def create_individual(self):
        path = list(range(self.n_cities))
        random.shuffle(path)
        return path
    


    def calculate_fitness(self, individual):
        dist = 0
        for i in range(self.n_cities):
            u = individual[i]
            v = individual[(i + 1) % self.n_cities]
            dist += self.matrix[u][v]
        
        if dist == 0: return float('inf')
        return 1 / dist



    def crossover(self, parent1, parent2):
        start, end = sorted(random.sample(range(self.n_cities), 2))
        
        def ox_create_child(p1, p2):
            child = [-1] * self.n_cities
            child[start:end] = p1[start:end]

            current_pos = end
            for i in range(self.n_cities):
                city = p2[(end + i) % self.n_cities]
                if city not in child:
                    if current_pos >= self.n_cities:
                        current_pos = 0
                    while child[current_pos] != -1:
                        current_pos += 1
                        if current_pos >= self.n_cities:
                            current_pos = 0
                    child[current_pos] = city
            return child

        c1 = ox_create_child(parent1, parent2)
        c2 = ox_create_child(parent2, parent1)
        return c1, c2



    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            i, j = sorted(random.sample(range(self.n_cities), 2))
            individual[i:j+1] = individual[i:j+1][::-1]
            
        if random.random() < self.local_search_prob:
            individual = self.apply_two_opt(individual)
            
        return individual



    def apply_two_opt(self, path):
        best_path = path[:]
        improved = True
        count = 0
        max_iterations = 50
        
        while improved and count < max_iterations:
            improved = False
            for i in range(1, self.n_cities - 2):
                for j in range(i + 1, self.n_cities):
                    if j - i == 1: continue
                    
                    u, v = best_path[i-1], best_path[i]
                    x, y = best_path[j], best_path[(j+1) % self.n_cities]
                    
                    current_dist = self.matrix[u][v] + self.matrix[x][y]
                    new_dist = self.matrix[u][x] + self.matrix[v][y]
                    
                    if new_dist < current_dist:
                        best_path[i:j+1] = best_path[i:j+1][::-1]
                        improved = True
                        count += 1
                        break
                if improved: break
                
        return best_path