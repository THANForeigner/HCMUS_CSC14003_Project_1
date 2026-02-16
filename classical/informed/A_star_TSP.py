import heapq
import sys

class AStarTSP:
    def __init__(self, distance_matrix):
        self.matrix = distance_matrix
        self.n = len(distance_matrix)
        
    def solve(self):
        start_node = 0
        initial_mask = 1 << start_node
        initial_path = [start_node]
        h = self._heuristic(start_node, initial_mask)
        pq = [(h, 0, start_node, initial_mask, initial_path)]
        best_g = {}
        
        while pq:
            f, g, u, mask, path = heapq.heappop(pq)
            if mask == (1 << self.n) - 1:
                total_cost = g + self.matrix[u][start_node]
                return total_cost, path
            if (u, mask) in best_g and best_g[(u, mask)] <= g:
                continue
            best_g[(u, mask)] = g
            for v in range(self.n):
                if not (mask & (1 << v)):
                    new_g = g + self.matrix[u][v]
                    new_mask = mask | (1 << v)
                    h = self._heuristic(v, new_mask)
                    new_f = new_g + h
                    new_path = path + [v]
                    heapq.heappush(pq, (new_f, new_g, v, new_mask, new_path))
                    
        return float('inf'), []
    def _heuristic(self, current_node, visited_mask):
        # MST heuristic for remaining unvisited nodes
        unvisited = []
        for i in range(self.n):
            if not (visited_mask & (1 << i)):
                unvisited.append(i)
        
        if not unvisited:
            return self.matrix[current_node][0] # Return to start cost

        if len(unvisited) == 1:
            return self.matrix[current_node][unvisited[0]] + self.matrix[unvisited[0]][0]
            
        mst_cost = 0
        if unvisited:
            key = {node: float('inf') for node in unvisited}
            key[unvisited[0]] = 0
            mst_set = set()
            
            while len(mst_set) < len(unvisited):
                # Find min key node
                u = min((node for node in unvisited if node not in mst_set), key=lambda k: key[k])
                mst_set.add(u)
                mst_cost += key[u]
                
                for v in unvisited:
                    if v not in mst_set:
                        if self.matrix[u][v] < key[v]:
                            key[v] = self.matrix[u][v]
                            
        # Plus minimum distance from current node to any unvisited
        min_to_unvisited = min(self.matrix[current_node][v] for v in unvisited)
        
        # Plus minimum distance from any unvisited to start node
        min_from_unvisited = min(self.matrix[v][0] for v in unvisited)
        
        return mst_cost + min_to_unvisited + min_from_unvisited
