import random

class HillClimbingTSP:
    def __init__(self, distance_matrix, max_iterations=1000, restarts=10):
        self.matrix = distance_matrix
        self.n = len(distance_matrix)
        self.max_iterations = max_iterations
        self.restarts = restarts

    def _calculate_total_distance(self, path):
        cost = 0.0
        if self.n == 0:
            return cost
        for i in range(self.n):
            u = path[i]
            v = path[(i + 1) % self.n]
            cost += self.matrix[u][v]
        return cost

    def _get_neighbors(self, path):
        neighbors = []
        for i in range(self.n - 1):
            for j in range(i + 1, self.n):
                neighbor = path[:]
                # 2-opt: reverse the sub-array between i and j
                neighbor[i:j+1] = reversed(neighbor[i:j+1])
                neighbors.append(neighbor)
        return neighbors

    def _hill_climb(self):
        current_path = list(range(self.n))
        random.shuffle(current_path)
        current_cost = self._calculate_total_distance(current_path)

        for _ in range(self.max_iterations):
            neighbors = self._get_neighbors(current_path)
            
            best_neighbor = None
            best_neighbor_cost = float('inf')
            
            for neighbor in neighbors:
                cost = self._calculate_total_distance(neighbor)
                if cost < best_neighbor_cost:
                    best_neighbor = neighbor
                    best_neighbor_cost = cost
                    
            if best_neighbor_cost >= current_cost:
                break
                
            current_path = best_neighbor
            current_cost = best_neighbor_cost
            
        return current_cost, current_path

    def solve(self):
        """
        Solves the Travelling Salesman Problem using Hill Climbing with random restarts.
        Returns:
            tuple: (best_cost, best_path)
        """
        if self.n <= 1:
            return 0.0, list(range(self.n))
            
        best_overall_cost = float('inf')
        best_overall_path = []
        
        for _ in range(self.restarts):
            cost, path = self._hill_climb()
            if cost < best_overall_cost:
                best_overall_cost = cost
                best_overall_path = path
                
        return best_overall_cost, best_overall_path
