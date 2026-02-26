import random
import math

class ACO_TSP:
    def __init__(self, distance_matrix, n_ants=10, max_iter=100, alpha=1.0, beta=2.0, evaporation_rate=0.5, Q=100.0):
        self.matrix = distance_matrix
        self.n_cities = len(distance_matrix)
        self.n_ants = n_ants
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.Q = Q
        self.pheromone = [[0.1 for _ in range(self.n_cities)] for _ in range(self.n_cities)]
        self.visibility = [[0.0 for _ in range(self.n_cities)] for _ in range(self.n_cities)]

        for i in range(self.n_cities):
            for j in range(self.n_cities):
                if i != j and self.matrix[i][j] > 0:
                    self.visibility[i][j] = 1.0 / self.matrix[i][j]
                else:
                    self.visibility[i][j] = 0.0

    def run(self):
        best_path = None
        best_cost = float('inf')
        
        for _ in range(self.max_iter):
            all_paths = []
            all_costs = []
            
            # Construct solutions for each ant
            for _ in range(self.n_ants):
                path, cost = self._construct_solution()
                all_paths.append(path)
                all_costs.append(cost)
                
                if cost < best_cost:
                    best_cost = cost
                    best_path = path[:]
            
            # Update pheromones
            self._update_pheromones(all_paths, all_costs)
            
        return best_cost, best_path

    def _construct_solution(self):
        start_node = random.randint(0, self.n_cities - 1)
        path = [start_node]
        visited = set([start_node])
        current = start_node
        cost = 0
        
        for _ in range(self.n_cities - 1):
            next_node = self._select_next_node(current, visited)
            path.append(next_node)
            visited.add(next_node)
            cost += self.matrix[current][next_node]
            current = next_node

        cost += self.matrix[current][start_node]
        path.append(start_node)
        return path, cost

    def _select_next_node(self, current, visited):
        probabilities = []
        total_prob = 0
        
        candidates = [i for i in range(self.n_cities) if i not in visited]
        
        for city in candidates:
            tau = self.pheromone[current][city] ** self.alpha
            eta = self.visibility[current][city] ** self.beta
            prob = tau * eta
            probabilities.append(prob)
            total_prob += prob
            
        if total_prob == 0:
            return random.choice(candidates)

        r = random.uniform(0, total_prob)
        cum_prob = 0
        for i, city in enumerate(candidates):
            cum_prob += probabilities[i]
            if r <= cum_prob:
                return city
        
        return candidates[-1]

    def _update_pheromones(self, paths, costs):
        # Evaporation
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                self.pheromone[i][j] *= (1.0 - self.evaporation_rate)
                
        # Deposit
        for path, cost in zip(paths, costs):
            deposit = self.Q / cost if cost > 0 else 0
            for i in range(self.n_cities - 1):
                u, v = path[i], path[i+1]
                self.pheromone[u][v] += deposit
                self.pheromone[v][u] += deposit
            
            # Don't forget the return edge
            u, v = path[-1], path[0]
            self.pheromone[u][v] += deposit
            self.pheromone[v][u] += deposit
